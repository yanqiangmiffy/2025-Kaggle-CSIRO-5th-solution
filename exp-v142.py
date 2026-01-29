"""
CSIRO Image2Biomass – Training Script
====================================

This module rewrites the cassava-starter training notebook into a python script
targeted at the CSIRO Biomass competition while reusing the two-stream,
three-head architecture from `example.py`.
"""

import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR, LinearLR, SequentialLR
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import r2_score, accuracy_score
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from utils import EMA


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CFG:
    """Configuration inspired by cassava notebook but tailored to csiro data."""

    seed: int = 42
    debug: bool = False
    n_fold: int = 5
    trn_fold: list[int] = field(default_factory=lambda: [0])
    model_name: str = "timm/vit_large_patch16_dinov3_qkvb.lvd1689m"
    img_size: int = 1024
    epochs: int = 20
    freeze_epochs: int = 2
    batch_size: int = 1
    num_workers: int = 0
    lr: float = 3e-4  # 提高冻结阶段学习率
    finetune_lr: float = 5e-5  # 提高微调阶段学习率
    min_lr: float = 1e-7  # 降低最小学习率
    weight_decay: float = 1e-6
    backbone_lr_mult: float = 0.1  # backbone学习率倍数
    head_lr_mult: float = 1.0  # head学习率倍数
    warmup_epochs: int = 2  # warmup epochs
    warmup_lr_init: float = 1e-6  # warmup初始学习率
    scheduler: Optional[str] = "CosineAnnealingWarmRestarts"
    scheduler_factor: float = 0.5  # 降低scheduler factor，使学习率衰减更平滑
    scheduler_patience: int = 6  # 增加patience
    scheduler_eps: float = 1e-6
    scheduler_T_max: int = 15  # 增加T_max
    scheduler_T_0: int = 15  # 增加T_0
    grad_accumulation_steps: int = 4
    max_grad_norm: float = 1_000.0
    apex: bool = False
    print_freq: int = 100
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_epochs:int=30
    early_stopping_patience:int=10

    MOE_EXPERTS: int = 4
    MOE_HIDDEN: int = 512
    MOE_DROPOUT: float = 0.2

    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "total_loss": 0.45,
        "gdm_loss": 0.2,
        "green_loss": 0.1,
        "dead_loss": 0.1,      # 物理约束：Dead = Total - GDM
        "clover_loss": 0.1,     # 物理约束：Clover = GDM - Green
        "ndvi_loss": 0.05,
        "height_loss": 0.05,
        "species_loss": 0.05,   # Species辅助分类任务
        "state_loss": 0.05,     # State辅助分类任务（澳大利亚州，隐含气候信息）
        "ratio_loss": 0.1,      # XG_mask ratio auxiliary heads
    })
    r2_weights: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1, 0.2, 0.5]))
    xg_mask_dir: Path = field(default_factory=lambda: Path("input/segment/output/SegVeg_XG_SVM_mask"))
    target_train: list[str] = field(default_factory=lambda: [
        "Dry_Total_g",
        "GDM_g",
        "Dry_Green_g",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ])
    target_eval: list[str] = field(
        default_factory=lambda: ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    )
    pretrained_path: str = '/data/pretrained_models/cv/timm/vit_large_patch16_dinov3_qkvb.lvd1689m/pytorch_model.bin'
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CSIRO_DATA_DIR", "./input/csiro-biomass")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "output/exp-v142")))
    resume_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.train_csv = self.data_dir / "train.csv"
        self.image_dir = self.data_dir / "train"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 验证 epochs 参数逻辑
        if self.freeze_epochs >= self.epochs:
            raise ValueError(f"freeze_epochs ({self.freeze_epochs}) must be < epochs ({self.epochs})")
        if self.ema_epochs < self.freeze_epochs:
            raise ValueError(f"ema_epochs ({self.ema_epochs}) must be >= freeze_epochs ({self.freeze_epochs})")
        # 如果 ema_epochs > epochs，自动调整为 epochs（避免 Stage 3 空循环）
        if self.ema_epochs > self.epochs:
            import warnings
            warnings.warn(f"ema_epochs ({self.ema_epochs}) > epochs ({self.epochs}), adjusting ema_epochs to {self.epochs}")
            self.ema_epochs = self.epochs
        # Species分类相关
        self.species_encoder: Optional[LabelEncoder] = None
        self.num_species: int = 0
        # State分类相关（澳大利亚州，隐含气候信息）
        self.state_encoder: Optional[LabelEncoder] = None
        self.num_states: int = 0

cfg = CFG()
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(cfg.seed)

# =============================================================================
# Logging / Utilities
# =============================================================================
LOGGER = logging.getLogger("CSIRO")



def init_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers = []
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)




@contextmanager
def timer(name: str):
    t0 = time.time()
    LOGGER.info("[%s] start", name)
    yield
    LOGGER.info("[%s] done in %.0f s", name, time.time() - t0)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


# =============================================================================
# Data preparation
# =============================================================================


def load_train_dataframe(cfg: CFG) -> pd.DataFrame:
    df_long = pd.read_csv(cfg.train_csv)
    df_wide = (
        df_long.pivot(index="image_path", columns="target_name", values="target")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # 注意：image_path 是连接键，必须包含
    meta_cols = ['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
    # 2. 针对 image_path 进行去重
    # 因为同一张图片的5行数据中，Date, State, NDVI 这些物理属性是完全一样的
    train_meta = df_long[meta_cols].drop_duplicates(subset=['image_path'])
    # 检查一下形状，应该是 (357, 6)
    print(f"Meta shape: {train_meta.shape}")
    # 3. 合并到 train_wide
    df_wide = df_wide.merge(train_meta, on='image_path', how='left')

    # 4. 对 Species 进行 Label Encoding
    # 处理缺失值，用空字符串填充
    df_wide['Species'] = df_wide['Species'].fillna('')
    cfg.species_encoder = LabelEncoder()
    df_wide['Species_encoded'] = cfg.species_encoder.fit_transform(df_wide['Species'].astype(str))
    cfg.num_species = len(cfg.species_encoder.classes_)
    LOGGER.info(cfg.species_encoder.classes_)
    LOGGER.info("Species encoding: %d unique species found", cfg.num_species)

    # 5. 对 State 进行 Label Encoding（澳大利亚州，隐含气候信息）
    df_wide['State'] = df_wide['State'].fillna('')
    cfg.state_encoder = LabelEncoder()
    df_wide['State_encoded'] = cfg.state_encoder.fit_transform(df_wide['State'].astype(str))
    cfg.num_states = len(cfg.state_encoder.classes_)
    LOGGER.info(cfg.state_encoder.classes_)

    LOGGER.info("State encoding: %d unique states found", cfg.num_states)

    # 5. 验证最终结果
    print(f"Final shape: {df_wide.shape}")
    print(f"Final columns: {df_wide.columns.tolist()}")
    print(df_wide)
    if cfg.debug:
        df_wide = df_wide.sample(n=200, random_state=cfg.seed).reset_index(drop=True)
    LOGGER.info("Loaded %d unique images", len(df_wide))

    return df_wide

def create_folds(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    自动搜索最佳种子，使得 StratifiedGroupKFold 分出的每个 Fold 大小尽可能均匀
    """
    df = df.copy()
    best_seed = cfg.seed
    best_std = float('inf')  # 记录最小的标准差
    best_df = None

    # 尝试搜索 50 个不同的种子
    print("Searching for best seed to balance folds...")
    for seed in range(cfg.seed, cfg.seed + 50):
        # print(f"Searching for best seed: {seed}")
        temp_df = df.copy()
        temp_df["fold"] = -1
        sgkf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=seed)
        groups = temp_df["Sampling_Date"]
        y = temp_df["State"]
        # 尝试划分
        for fold, (_, val_idx) in enumerate(sgkf.split(temp_df, y, groups=groups)):
            temp_df.loc[val_idx, "fold"] = fold
        # 计算 Fold 大小的标准差（越小越均衡）
        fold_counts = temp_df["fold"].value_counts()
        current_std = fold_counts.std()
        if current_std < best_std:
            best_std = current_std
            best_seed = seed
            best_df = temp_df.copy()
            print(f"Found better seed: {seed}, Std: {current_std:.2f}, Counts: {fold_counts.tolist()}")

    print(f"Best Seed Found: {best_seed}")
    return best_df



# =============================================================================
# Augmentations
# =============================================================================

def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.75),

            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_valid_transforms(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# =============================================================================
# Dataset
# =============================================================================
def clean_image(img):
    """
    图像预处理：移除底部伪影和日期戳
    
    Args:
        img: RGB 格式的图像 (numpy array)
    
    Returns:
        处理后的图像
    """
    # 1. Safe Crop (Remove artifacts at the bottom)
    h, w = img.shape[:2]
    # Cut bottom 10% where artifacts often appear
    # img = img[0:int(h*0.95), :] 

    # 2. Inpaint Date Stamp (Remove orange text)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define orange color range (adjust as needed)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Dilate mask to cover text edges and reduce noise
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)

    # Inpaint if mask is not empty
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return img

class BiomassDataset(Dataset):
    """Three-stream dataset that splits 2000x1000 image into halves plus full image."""

    def __init__(
            self,
            df: pd.DataFrame,
            transforms_fn: Callable[[], A.Compose],
            image_dir: Path,
            train_target_cols: list[str],
            all_target_cols: list[str],
    ):
        self.df = df.reset_index(drop=True)
        self.transforms_fn = transforms_fn
        self.image_dir = image_dir
        self.train_targets = self.df[train_target_cols].values.astype(np.float32)
        self.eval_targets = self.df[all_target_cols].values.astype(np.float32)
        self.species_targets = self.df["Species_encoded"].values.astype(np.int64)
        self.image_paths = self.df["image_path"].values

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        full_path = self.image_dir / Path(path).name
        img = cv2.imread(str(full_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load {full_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int):
        image = self._load_image(self.image_paths[idx])
        image=clean_image(image)
        
        height, width = image.shape[:2]
        mid = width // 2
        left = image[:, :mid]
        right = image[:, mid:]
        full_image = image.copy()

        transform_left = self.transforms_fn()
        transform_right = self.transforms_fn()
        transform_full = self.transforms_fn()
        left = transform_left(image=left)["image"]
        right = transform_right(image=right)["image"]
        full_image = transform_full(image=full_image)["image"]

        train_tgt = torch.tensor(self.train_targets[idx])
        eval_tgt = torch.tensor(self.eval_targets[idx])
        species_tgt = torch.tensor(self.species_targets[idx])
        return left, right, full_image, train_tgt, eval_tgt, species_tgt


class BiomassDatasetWithRatiosAndSpecies(Dataset):
    """Dataset that includes Species, State targets and XG_mask class ratios as auxiliary targets.
    
    XG_SVM mask has 3 classes:
    - 0: soil
    - 1: healthy vegetation
    - 2: chlorosis (senescent)
    
    Returns ratios for each class as additional training targets, along with species and state labels.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            transforms_fn: Callable[[], A.Compose],
            image_dir: Path,
            train_target_cols: list[str],
            all_target_cols: list[str],
            xg_mask_dir: Path,
    ):
        self.df = df.reset_index(drop=True)
        self.transforms_fn = transforms_fn
        self.image_dir = image_dir
        self.train_targets = self.df[train_target_cols].values.astype(np.float32)
        self.eval_targets = self.df[all_target_cols].values.astype(np.float32)
        self.species_targets = self.df["Species_encoded"].values.astype(np.int64)
        self.state_targets = self.df["State_encoded"].values.astype(np.int64)
        self.image_paths = self.df["image_path"].values
        self.xg_mask_dir = xg_mask_dir

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        full_path = self.image_dir / Path(path).name
        img = cv2.imread(str(full_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load {full_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_xg_ratios(self, image_path: str) -> np.ndarray:
        """Load XG_SVM mask and compute class ratios.
        
        Returns:
            np.ndarray: [soil_ratio, healthy_ratio, chlorosis_ratio]
        """
        # Extract image ID from path (e.g., "train/ID1012260530.jpg" -> "ID1012260530")
        basename = Path(image_path).stem
        mask_path = self.xg_mask_dir / f"{basename}_xg_mask.npy"
        
        if not mask_path.exists():
            # Return NaN if mask not found
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        
        mask = np.load(mask_path)
        total_pixels = mask.size
        
        # XG_SVM: 0=soil, 1=healthy, 2=chlorosis
        ratios = np.zeros(3, dtype=np.float32)
        for cls in [0, 1, 2]:
            ratios[cls] = np.sum(mask == cls) / total_pixels
        
        return ratios

    def __getitem__(self, idx: int):
        image = self._load_image(self.image_paths[idx])
        image = clean_image(image)
        
        height, width = image.shape[:2]
        mid = width // 2
        left = image[:, :mid]
        right = image[:, mid:]
        full_image = image.copy()

        transform_left = self.transforms_fn()
        transform_right = self.transforms_fn()
        transform_full = self.transforms_fn()
        left = transform_left(image=left)["image"]
        right = transform_right(image=right)["image"]
        full_image = transform_full(image=full_image)["image"]

        train_tgt = torch.tensor(self.train_targets[idx])
        eval_tgt = torch.tensor(self.eval_targets[idx])
        species_tgt = torch.tensor(self.species_targets[idx])
        state_tgt = torch.tensor(self.state_targets[idx])
        xg_ratios = torch.tensor(self._load_xg_ratios(self.image_paths[idx]))
        
        return left, right, full_image, train_tgt, eval_tgt, species_tgt, state_tgt, xg_ratios


# =============================================================================
# Model + Loss + Metric
# =============================================================================

class MetaMoE(nn.Module):
    """Mixture-of-Experts head used for main targets.
    
    Now outputs both mean and variance for GaussianNLLLoss.
    """

    def __init__(self, in_dim: int, hidden: int, num_experts: int, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
        )
        # Output 2 values: mean and variance_logit
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2),  # Output [mean, var_logit]
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1).unsqueeze(-1)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        out = (expert_outs * gate_weights).sum(dim=1)  # [B, 2]
        
        pred_mean = out[:, 0:1]  # [B, 1]
        pred_var_logit = out[:, 1:2]  # [B, 1]
        # Use softplus to ensure variance is positive, add small epsilon for numerical stability
        pred_var = F.softplus(pred_var_logit) + 1e-6
        
        return pred_mean, pred_var


class BiomassModel(nn.Module):
    """Shared backbone with regression heads, species/state classification heads, and XG_mask ratio heads."""

    def __init__(self, model_name: str, pretrained: bool = True, pretrained_path: str = '', num_species: int = 0, num_states: int = 0):
        super().__init__()
        if pretrained_path:
            self.backbone=timm.create_model(
                model_name,
                pretrained=True,
                pretrained_cfg_overlay=dict(file=pretrained_path),
                num_classes=0,
                img_size=CFG.img_size
            )
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.n_features = self.backbone.num_features * 3
        self.head_total = MetaMoE(
            in_dim=self.n_features,
            hidden=CFG.MOE_HIDDEN,
            num_experts=CFG.MOE_EXPERTS,
            dropout=CFG.MOE_DROPOUT,
            out_dim=1,
        )
        self.head_gdm = MetaMoE(
            in_dim=self.n_features,
            hidden=CFG.MOE_HIDDEN,
            num_experts=CFG.MOE_EXPERTS,
            dropout=CFG.MOE_DROPOUT,
            out_dim=1,
        )
        self.head_green = MetaMoE(
            in_dim=self.n_features,
            hidden=CFG.MOE_HIDDEN,
            num_experts=CFG.MOE_EXPERTS,
            dropout=CFG.MOE_DROPOUT,
            out_dim=1,
        )
        self.head_ndvi = self._make_head()
        self.head_height = self._make_head()
        
        # Classification head for species (auxiliary task)
        if num_species > 0:
            self.head_species = self._make_classification_head(num_species)
        else:
            self.head_species = None
        
        # Classification head for state (auxiliary task - Australian states, implicit climate info)
        if num_states > 0:
            self.head_state = self._make_classification_head(num_states)
        else:
            self.head_state = None
        
        # XG_mask ratio prediction heads (auxiliary)
        self.head_soil_ratio = self._make_head()
        self.head_healthy_ratio = self._make_head()
        self.head_chlorosis_ratio = self._make_head()

    def _make_head(self) -> nn.Module:
        """Create a head that outputs (mean, variance) for auxiliary targets."""
        hidden = self.n_features // 2
        return _AuxHead(self.n_features, hidden)
    
    def _make_classification_head(self, num_classes: int) -> nn.Sequential:
        """Create a classification head for species/state prediction."""
        hidden = self.n_features // 2
        return nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, full_image: torch.Tensor):
        feat_left = self.backbone(left)
        feat_right = self.backbone(right)
        feat_full = self.backbone(full_image)
        feats = torch.cat([feat_left, feat_right, feat_full], dim=1)
        # Each head returns (mean, var) tuple
        total_mean, total_var = self.head_total(feats)
        gdm_mean, gdm_var = self.head_gdm(feats)
        green_mean, green_var = self.head_green(feats)
        ndvi_mean, ndvi_var = self.head_ndvi(feats)
        height_mean, height_var = self.head_height(feats)
        
        # XG_mask ratio predictions
        soil_ratio_mean, soil_ratio_var = self.head_soil_ratio(feats)
        healthy_ratio_mean, healthy_ratio_var = self.head_healthy_ratio(feats)
        chlorosis_ratio_mean, chlorosis_ratio_var = self.head_chlorosis_ratio(feats)
        
        # Get species logits if available
        species_logits = None
        if self.head_species is not None:
            species_logits = self.head_species(feats)
        
        # Get state logits if available
        state_logits = None
        if self.head_state is not None:
            state_logits = self.head_state(feats)
        
        return (
            (total_mean, total_var),
            (gdm_mean, gdm_var),
            (green_mean, green_var),
            (ndvi_mean, ndvi_var),
            (height_mean, height_var),
            # Ratio predictions
            (soil_ratio_mean, soil_ratio_var),
            (healthy_ratio_mean, healthy_ratio_var),
            (chlorosis_ratio_mean, chlorosis_ratio_var),
            # Classification predictions (can be None)
            species_logits,
            state_logits,
        )


class _AuxHead(nn.Module):
    """Auxiliary head for NDVI and Height that outputs (mean, variance)."""
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, 2)  # Output [mean, var_logit]
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        pred_mean = out[:, 0:1]
        pred_var_logit = out[:, 1:2]
        pred_var = F.softplus(pred_var_logit) + 1e-6
        return pred_mean, pred_var


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics-consistent loss using GaussianNLLLoss with species/state classification and XG_mask ratio auxiliary tasks.
    
    Key benefits:
    - Automatic outlier handling: model learns to predict high variance for noisy samples
    - Handles heteroscedasticity: different uncertainty for different biomass levels
    - Dead = Total - GDM
    - Clover = GDM - Green
    - Species classification as auxiliary task for better feature learning
    - State classification as auxiliary task (Australian states, implicit climate info)
    - XG_mask ratio prediction as auxiliary task for vegetation understanding
    """
    def __init__(self, weights: dict[str, float] = None):
        super().__init__()
        # 如果没有指定，使用默认权重
        if weights is None:
            self.w = {
                "total": 0.45,
                "gdm": 0.2,
                "green": 0.1,
                "dead": 0.1,
                "clover": 0.1,
                "ndvi": 0.0,
                "height": 0.0,
                "species": 0.05,
                "state": 0.05,
                "ratio": 0.1,
            }
        else:
            # 从配置中映射权重，如果没有则使用默认值
            self.w = {
                "total": weights.get("total_loss", 0.45),
                "gdm": weights.get("gdm_loss", 0.2),
                "green": weights.get("green_loss", 0.1),
                "dead": weights.get("dead_loss", 0.1),
                "clover": weights.get("clover_loss", 0.1),
                "ndvi": weights.get("ndvi_loss", 0.05),
                "height": weights.get("height_loss", 0.05),
                "species": weights.get("species_loss", 0.05),
                "state": weights.get("state_loss", 0.05),
                "ratio": weights.get("ratio_loss", 0.1),
            }

        # GaussianNLLLoss: automatically handles outliers by learning per-sample variance
        self.criterion = nn.GaussianNLLLoss()
        # Fallback for physics consistency (using means only)
        self.criterion_aux = nn.SmoothL1Loss(beta=1.0)
        # Classification loss for species and state
        self.classification_criterion = nn.CrossEntropyLoss()

    def forward(self, preds, targets, species_targets=None, state_targets=None, xg_ratios=None):
        """
        preds: tuple of 10 elements:
               ((total_mean, total_var), (gdm_mean, gdm_var), (green_mean, green_var),
                (ndvi_mean, ndvi_var), (height_mean, height_var),
                (soil_ratio_mean, soil_ratio_var), (healthy_ratio_mean, healthy_ratio_var),
                (chlorosis_ratio_mean, chlorosis_ratio_var), species_logits, state_logits)
        targets: tensor of shape [Batch, 5] -> [Total, GDM, Green, NDVI, Height]
                 注意：这里假设传入的 targets 顺序必须与 cfg.target_train 一致
        species_targets: optional tensor of shape [Batch] with species class indices
        state_targets: optional tensor of shape [Batch] with state class indices
        xg_ratios: optional tensor of shape [Batch, 3] -> [soil_ratio, healthy_ratio, chlorosis_ratio]
        """
        # Unpack all predictions (10 elements: 5 regression + 3 ratio + species_logits + state_logits)
        (pred_total_mean, pred_total_var), \
        (pred_gdm_mean, pred_gdm_var), \
        (pred_green_mean, pred_green_var), \
        (pred_ndvi_mean, pred_ndvi_var), \
        (pred_height_mean, pred_height_var), \
        (pred_soil_ratio_mean, pred_soil_ratio_var), \
        (pred_healthy_ratio_mean, pred_healthy_ratio_var), \
        (pred_chlorosis_ratio_mean, pred_chlorosis_ratio_var), \
        species_logits, state_logits = preds
        
        gt_total = targets[:, 0:1]
        gt_gdm = targets[:, 1:2]
        gt_green = targets[:, 2:3]
        gt_ndvi = targets[:, 3:4]
        gt_height = targets[:, 4:5]

        # --- 1. 直接预测的 GaussianNLL Loss ---
        # GaussianNLLLoss: -log(p(target | mean, var)) = 0.5 * (log(var) + (target - mean)^2 / var)
        loss_total = self.criterion(pred_total_mean, gt_total, pred_total_var)
        loss_gdm = self.criterion(pred_gdm_mean, gt_gdm, pred_gdm_var)
        loss_green = self.criterion(pred_green_mean, gt_green, pred_green_var)
        loss_ndvi = self.criterion(pred_ndvi_mean, gt_ndvi, pred_ndvi_var)
        loss_height = self.criterion(pred_height_mean, gt_height, pred_height_var)

        # --- 2. 物理约束/间接推导的 Loss (使用 mean 值) ---
        # 物理关系 1: Dead = Total - GDM
        # 对于物理约束，使用组合方差 var_dead = var_total + var_gdm
        pred_dead_mean = pred_total_mean - pred_gdm_mean
        pred_dead_var = pred_total_var + pred_gdm_var  # Variance adds for independent variables
        gt_dead = gt_total - gt_gdm 
        loss_dead = self.criterion(pred_dead_mean, gt_dead, pred_dead_var)

        # 物理关系 2: Clover = GDM - Green
        pred_clover_mean = pred_gdm_mean - pred_green_mean
        pred_clover_var = pred_gdm_var + pred_green_var
        gt_clover = gt_gdm - gt_green
        loss_clover = self.criterion(pred_clover_mean, gt_clover, pred_clover_var)

        # --- 3. 总 Loss ---
        total_loss = (
            self.w["total"] * loss_total +
            self.w["gdm"] * loss_gdm +
            self.w["green"] * loss_green +
            self.w["dead"] * loss_dead +      # 增加对 Dead 的约束
            self.w["clover"] * loss_clover +  # 增加对 Clover 的约束
            self.w["ndvi"] * loss_ndvi +      # NDVI GaussianNLL 损失
            self.w["height"] * loss_height    # Height GaussianNLL 损失
        )
        
        # --- 4. Species classification loss (if available) ---
        if species_logits is not None and species_targets is not None:
            loss_species = self.classification_criterion(species_logits, species_targets)
            total_loss += self.w["species"] * loss_species
        
        # --- 5. State classification loss (if available) ---
        if state_logits is not None and state_targets is not None:
            loss_state = self.classification_criterion(state_logits, state_targets)
            total_loss += self.w["state"] * loss_state
        
        # --- 6. XG_mask Ratio Loss (辅助监督) ---
        if xg_ratios is not None and self.w["ratio"] > 0:
            gt_soil_ratio = xg_ratios[:, 0:1]
            gt_healthy_ratio = xg_ratios[:, 1:2]
            gt_chlorosis_ratio = xg_ratios[:, 2:3]
            
            # Filter out NaN values (missing masks)
            valid_mask = ~torch.isnan(gt_soil_ratio).squeeze()
            if valid_mask.any():
                loss_soil_ratio = self.criterion(
                    pred_soil_ratio_mean[valid_mask], 
                    gt_soil_ratio[valid_mask], 
                    pred_soil_ratio_var[valid_mask]
                )
                loss_healthy_ratio = self.criterion(
                    pred_healthy_ratio_mean[valid_mask], 
                    gt_healthy_ratio[valid_mask], 
                    pred_healthy_ratio_var[valid_mask]
                )
                loss_chlorosis_ratio = self.criterion(
                    pred_chlorosis_ratio_mean[valid_mask], 
                    gt_chlorosis_ratio[valid_mask], 
                    pred_chlorosis_ratio_var[valid_mask]
                )
                ratio_loss = loss_soil_ratio + loss_healthy_ratio + loss_chlorosis_ratio
                total_loss = total_loss + self.w["ratio"] * ratio_loss
        
        return total_loss

def weighted_r2(preds: dict[str, np.ndarray], targets: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """
    计算加权 R2 分数，使用 sklearn 的 r2_score。
    
    Args:
        preds: 字典，包含 'total', 'gdm', 'green' 的预测值
        targets: 真实值数组，shape (N, 5)，列顺序为 [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
        weights: 权重数组，shape (5,)，对应每个目标的权重
    
    Returns:
        tuple: (weighted_score, r2_scores)
            - weighted_score: 加权 R2 分数（float）
            - r2_scores: 每个目标的 R2 分数数组，shape (5,)
    """
    pred_total = preds["total"]
    pred_gdm = preds["gdm"]
    pred_green = preds["green"]
    pred_dead = np.maximum(0, pred_total - pred_gdm)
    pred_clover = np.maximum(0, pred_gdm - pred_green)
    y_pred = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
    r2_scores = r2_score(targets, y_pred, multioutput="raw_values")
    weighted_score = float(np.sum(r2_scores * weights))
    return weighted_score, r2_scores




def format_r2_scores(r2_scores: np.ndarray, weights: np.ndarray) -> str:
    """
    格式化 R2 分数输出，用于日志记录。
    
    Args:
        r2_scores: 每个目标的 R2 分数数组，shape (5,)
        weights: 权重数组，shape (5,)
    
    Returns:
        格式化的字符串
    """
    target_names = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    parts = []
    for name, r2, weight in zip(target_names, r2_scores, weights):
        weighted_contrib = r2 * weight
        parts.append(f"{name}:{r2:.4f}(w={weight:.1f},c={weighted_contrib:.4f})")
    return " | ".join(parts)

    

def get_scheduler(cfg: CFG, optimizer: optim.Optimizer):
    """Build LR scheduler consistent with cassava template."""
    if cfg.scheduler is None:
        return None
    if cfg.scheduler == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            verbose=True,
            eps=cfg.scheduler_eps,
        )
    if cfg.scheduler == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=cfg.scheduler_T_max, eta_min=cfg.min_lr, last_epoch=-1)
    if cfg.scheduler == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler_T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1)
    LOGGER.warning("Unknown scheduler %s, disabling.", cfg.scheduler)
    return None


def step_scheduler(scheduler, val_metric: Optional[float] = None) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_metric)
    else:
        scheduler.step()


# =============================================================================
# Training helpers
# =============================================================================

def train_one_epoch(
        cfg: CFG,
        loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
        ema: Optional[EMA] = None,
) -> float:
    losses = AverageMeter()
    model.train()
    optimizer.zero_grad()
    
    # 计算实际的更新步数
    total_steps = len(loader)
    update_steps = (total_steps + cfg.grad_accumulation_steps - 1) // cfg.grad_accumulation_steps
    
    pbar = tqdm(enumerate(loader), total=total_steps, desc="train", leave=False)
    
    for step, (left, right, full_image, targets, _, species_targets, state_targets, xg_ratios) in pbar:
        left = left.to(device)
        right = right.to(device)
        full_image = full_image.to(device)
        targets = targets.to(device)
        species_targets = species_targets.to(device)
        state_targets = state_targets.to(device)
        xg_ratios = xg_ratios.to(device)
        
        with autocast(device_type='cuda', enabled=cfg.apex):
            preds = model(left, right, full_image)
            loss = criterion(preds, targets, species_targets, state_targets, xg_ratios)
            loss = loss / cfg.grad_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % cfg.grad_accumulation_steps == 0:
            current_update = (step + 1) // cfg.grad_accumulation_steps
            
            if cfg.max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if ema is not None:
                ema.update()
            
            # 显示当前更新步数和总更新步数
            pbar.set_postfix(
                loss=f"{losses.avg:.4f}",
                update=f"{current_update}/{update_steps}"
            )
        
        losses.update(loss.item() * cfg.grad_accumulation_steps, left.size(0))
    
    return losses.avg



def valid_one_epoch(
        cfg: CFG,
        loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        ema: Optional[EMA] = None,
) -> tuple[float, float, np.ndarray, float, float]:
    """
    验证一个 epoch，返回损失、加权 R2 分数、每个目标的 R2 分数以及分类准确率。
    
    Returns:
        tuple: (val_loss, score, r2_scores, species_acc, state_acc)
            - val_loss: 验证损失
            - score: 加权 R2 分数（用于 best_score 判断）
            - r2_scores: 每个目标的 R2 分数数组（用于详细分析）
            - species_acc: Species 分类准确率
            - state_acc: State 分类准确率
    """
    losses = AverageMeter()
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    preds_total, preds_gdm, preds_green = [], [], []
    targets_eval = []
    # For classification accuracy
    all_species_preds, all_species_targets = [], []
    all_state_preds, all_state_targets = [], []

    try:
        with torch.no_grad():
            pbar = tqdm(loader, desc="valid", leave=False)
            for left, right, full_image, train_targets, eval_targets, species_targets, state_targets, xg_ratios in pbar:
                left = left.to(device)
                right = right.to(device)
                full_image = full_image.to(device)
                train_targets = train_targets.to(device)
                species_targets = species_targets.to(device)
                state_targets = state_targets.to(device)
                xg_ratios = xg_ratios.to(device)

                with autocast(device_type='cuda', enabled=cfg.apex):
                    # Model returns 10 elements: 5 regression heads + 3 ratio heads + species_logits + state_logits
                    preds = model(left, right, full_image)
                    loss = criterion(preds, train_targets, species_targets, state_targets, xg_ratios)

                # Extract mean values for R2 calculation (first 3 heads only)
                (pred_total_mean, _), (pred_gdm_mean, _), (pred_green_mean, _), _, _, _, _, _, species_logits, state_logits = preds
                
                losses.update(loss.item(), left.size(0))
                preds_total.append(pred_total_mean.detach().cpu().numpy())
                preds_gdm.append(pred_gdm_mean.detach().cpu().numpy())
                preds_green.append(pred_green_mean.detach().cpu().numpy())
                targets_eval.append(eval_targets.numpy())
                
                # Collect classification predictions and targets
                if species_logits is not None:
                    species_pred = species_logits.argmax(dim=1).detach().cpu().numpy()
                    all_species_preds.append(species_pred)
                    all_species_targets.append(species_targets.detach().cpu().numpy())
                
                if state_logits is not None:
                    state_pred = state_logits.argmax(dim=1).detach().cpu().numpy()
                    all_state_preds.append(state_pred)
                    all_state_targets.append(state_targets.detach().cpu().numpy())
    finally:
        if ema is not None:
            ema.restore()

    preds = {
        "total": np.vstack(preds_total).flatten(),
        "gdm": np.vstack(preds_gdm).flatten(),
        "green": np.vstack(preds_green).flatten(),
    }
    targets = np.vstack(targets_eval)
    
    # 使用 weighted_r2 计算分数和详细的 R2 分数
    score, r2_scores = weighted_r2(preds, targets, cfg.r2_weights)
    
    # 计算分类准确率
    species_acc = 0.0
    if all_species_preds:
        all_species_preds = np.concatenate(all_species_preds)
        all_species_targets = np.concatenate(all_species_targets)
        species_acc = accuracy_score(all_species_targets, all_species_preds)
    
    state_acc = 0.0
    if all_state_preds:
        all_state_preds = np.concatenate(all_state_preds)
        all_state_targets = np.concatenate(all_state_targets)
        state_acc = accuracy_score(all_state_targets, all_state_preds)
    
    return losses.avg, score, r2_scores, species_acc, state_acc


# =============================================================================
# Two-stage trainer
# =============================================================================


def run_fold(cfg: CFG, df: pd.DataFrame, fold: int, device: torch.device) -> tuple[float, np.ndarray]:
    LOGGER.info("========== Fold %d ==========", fold)
    train_df = df
    valid_df = df

    # Use BiomassDatasetWithRatiosAndSpecies to include both species and XG_mask class ratios
    train_dataset = BiomassDatasetWithRatiosAndSpecies(
        train_df, lambda: get_train_transforms(cfg.img_size), cfg.image_dir, 
        cfg.target_train, cfg.target_eval, cfg.xg_mask_dir
    )
    valid_dataset = BiomassDatasetWithRatiosAndSpecies(
        valid_df, lambda: get_valid_transforms(cfg.img_size), cfg.image_dir, 
        cfg.target_train, cfg.target_eval, cfg.xg_mask_dir
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = BiomassModel(cfg.model_name, pretrained=True, pretrained_path=cfg.pretrained_path, num_species=cfg.num_species, num_states=cfg.num_states).to(device)
    # timm.create_model(name, pretrained=pretrained, pretrained_cfg_overlay=dict(file=CFG.pretrained_path), num_classes=0)
    criterion = PhysicsConsistencyLoss(cfg.loss_weights).to(device)
    scaler = GradScaler(enabled=cfg.apex)

    ema = None
    if cfg.use_ema:
        ema = EMA(model, cfg.ema_decay)
        ema.register()

    best_score = -np.inf
    best_path = cfg.output_dir / f"fold{fold}_best.pth"

    # Stage 1: 冻结backbone，只训练head
    LOGGER.info("Stage 1: freezing backbone for %d epochs", cfg.freeze_epochs)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 分层学习率：head使用较高学习率
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'head' in name:
                head_params.append({'params': param, 'lr': cfg.lr * cfg.head_lr_mult})
            else:
                head_params.append({'params': param, 'lr': cfg.lr})

    optimizer = optim.AdamW(head_params, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(cfg, optimizer)
    LOGGER.info("Stage 1 LR: base=%.2e, head_mult=%.1f, min=%.2e, warmup=%d epochs",
                cfg.lr, cfg.head_lr_mult, cfg.min_lr, cfg.warmup_epochs)

    # Warmup调度器包装
    if cfg.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.warmup_lr_init / cfg.lr,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[cfg.warmup_epochs]
        )
    epochs_without_improvement=0
    for epoch in range(1, cfg.freeze_epochs + 1):
        LOGGER.info("Fold %d Epoch %d/%d (frozen)", fold, epoch, cfg.epochs)
        train_loss = train_one_epoch(cfg, train_loader, model, criterion, optimizer, scaler, device)
        LOGGER.info("Fold %d Epoch %d -> train_loss %.4f", fold, epoch, train_loss)
        val_loss, score, r2_scores, species_acc, state_acc = valid_one_epoch(cfg, valid_loader, model, criterion, device)
        step_scheduler(scheduler, val_loss)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement=0
        else:
            epochs_without_improvement+=1

        if epochs_without_improvement >= CFG.early_stopping_patience:
            print(f"早停: epoch={epoch}, {CFG.early_stopping_patience} 个 epoch 无改善")
            break

        LOGGER.info("Fold %d Epoch %d -> val_loss %.4f | score %.4f | best_score %.4f", fold, epoch, val_loss, score, best_score)
        LOGGER.info("  R2 Scores: %s", format_r2_scores(r2_scores, cfg.r2_weights))
        LOGGER.info("  Classification Acc: species=%.4f, state=%.4f", species_acc, state_acc)

    # Stage 2：微调整个模型，使用分层学习率
    LOGGER.info("Stage 2: full fine-tuning for remaining epochs")
    for param in model.backbone.parameters():
        param.requires_grad = True

    # 分层学习率：backbone使用较低学习率，head使用较高学习率
    param_groups = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param_groups.append({'params': param, 'lr': cfg.finetune_lr * cfg.backbone_lr_mult})
        elif 'head' in name:
            param_groups.append({'params': param, 'lr': cfg.finetune_lr * cfg.head_lr_mult})
        else:
            param_groups.append({'params': param, 'lr': cfg.finetune_lr})

    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(cfg, optimizer)
    LOGGER.info("Stage 2 LR: finetune=%.2e, backbone_mult=%.1f, head_mult=%.1f, min=%.2e",
                cfg.finetune_lr, cfg.backbone_lr_mult, cfg.head_lr_mult, cfg.min_lr)
    # 确保 Stage 2 不超过总 epochs 数
    stage2_end = min(cfg.ema_epochs + 1, cfg.epochs + 1)
    for epoch in range(cfg.freeze_epochs + 1, stage2_end):
        LOGGER.info("Fold %d Epoch %d/%d (finetune)", fold, epoch, cfg.epochs)
        train_loss = train_one_epoch(cfg, train_loader, model, criterion, optimizer, scaler, device)
        LOGGER.info("Fold %d Epoch %d -> train_loss %.4f", fold, epoch, train_loss)
        val_loss, score, r2_scores, species_acc, state_acc = valid_one_epoch(cfg, valid_loader, model, criterion, device)
        step_scheduler(scheduler, val_loss)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= CFG.early_stopping_patience:
            print(f"早停: epoch={epoch}, {CFG.early_stopping_patience} 个 epoch 无改善")
            break
        LOGGER.info("Fold %d Epoch %d -> val_loss %.4f | score %.4f | best_score %.4f", fold, epoch, val_loss, score, best_score)
        LOGGER.info("  R2 Scores: %s", format_r2_scores(r2_scores, cfg.r2_weights))
        LOGGER.info("  Classification Acc: species=%.4f, state=%.4f", species_acc, state_acc)

    # Stage 3：EMA (只有当 ema_epochs < epochs 时才执行)
    if cfg.ema_epochs < cfg.epochs:
        for epoch in range(cfg.ema_epochs + 1, cfg.epochs + 1):
            if cfg.use_ema:
                LOGGER.info("Fold %d Epoch %d/%d (ema)", fold, epoch, cfg.epochs)
            else:
                LOGGER.info("Fold %d Epoch %d/%d (finetune)", fold, epoch, cfg.epochs)
            train_loss = train_one_epoch(cfg, train_loader, model, criterion, optimizer, scaler, device,ema)
            LOGGER.info("Fold %d Epoch %d -> train_loss %.4f", fold, epoch, train_loss)
            val_loss, score, r2_scores, species_acc, state_acc = valid_one_epoch(cfg, valid_loader, model, criterion, device, ema)
            step_scheduler(scheduler, val_loss)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), best_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= CFG.early_stopping_patience:
                print(f"早停: epoch={epoch}, {CFG.early_stopping_patience} 个 epoch 无改善")
                break
            LOGGER.info("Fold %d Epoch %d -> val_loss %.4f | score %.4f | best_score %.4f", fold, epoch, val_loss, score, best_score)
            LOGGER.info("  R2 Scores: %s", format_r2_scores(r2_scores, cfg.r2_weights))
            LOGGER.info("  Classification Acc: species=%.4f, state=%.4f", species_acc, state_acc)
    else:
        LOGGER.info("Skipping Stage 3 (EMA): ema_epochs (%d) >= epochs (%d)", cfg.ema_epochs, cfg.epochs)
    
    # 加载最佳模型并计算最终的详细 R2 分数
    model.load_state_dict(torch.load(best_path))
    _, final_score, final_r2_scores, final_species_acc, final_state_acc = valid_one_epoch(cfg, valid_loader, model, criterion, device)
    LOGGER.info("Fold %d Final Best Model - weighted_r2 score: %.4f", fold, final_score)
    LOGGER.info("Fold %d Final Best Model R2 Scores (detailed): %s", fold, format_r2_scores(final_r2_scores, cfg.r2_weights))
    LOGGER.info("Fold %d Final Best Model Classification Acc: species=%.4f, state=%.4f", fold, final_species_acc, final_state_acc)
    
    return best_score, final_r2_scores


# =============================================================================
# Main
# =============================================================================


def main():
    

    init_logger(cfg.output_dir / "train.log")

    LOGGER.info("Config: %s", cfg)
    df = create_folds(load_train_dataframe(cfg), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    all_r2_scores = []  # 存储每个 fold 的详细 R2 分数
    with timer("training"):
        for fold in cfg.trn_fold:
            score, r2_scores = run_fold(cfg, df, fold, device)
            scores.append(score)
            all_r2_scores.append(r2_scores)
            LOGGER.info("Fold %d best score %.4f", fold, score)
            LOGGER.info("Fold %d R2 Scores: %s", fold, format_r2_scores(r2_scores, cfg.r2_weights))

    # 计算平均分数和详细统计
    avg_score = np.mean(scores)
    LOGGER.info("CV score %.4f +/- %.4f", avg_score, np.std(scores))
    LOGGER.info("Individual fold scores: %s", scores)
    
    # 计算每个目标的平均 R2 分数
    all_r2_scores = np.array(all_r2_scores)  # shape: (n_folds, 5)
    target_names = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    LOGGER.info("=" * 70)
    LOGGER.info("Average R2 Scores across all folds:")
    for i, name in enumerate(target_names):
        mean_r2 = np.mean(all_r2_scores[:, i])
        std_r2 = np.std(all_r2_scores[:, i])
        weight = cfg.r2_weights[i]
        LOGGER.info("  %s: %.4f +/- %.4f (weight=%.1f)", name, mean_r2, std_r2, weight)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
