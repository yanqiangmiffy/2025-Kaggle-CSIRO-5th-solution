# ====================================================
# Library
# ====================================================
import os
import gc
import time
import random
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass, field
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import warnings

warnings.filterwarnings('ignore')


# ====================================================
# Config
# ====================================================
@dataclass
class GlobalConfig:
    """Global configuration for inference."""
    output_dir: str = './'
    data_dir: Path = Path(os.environ.get("CSIRO_DATA_DIR", "/kaggle/input/csiro-biomass"))
    seed: int = 42
    batch_size: int = 2
    num_workers: int = 0
    img_size: int = 1024
    use_tta: bool = True

    # Devices
    device_0: torch.device = field(default_factory=lambda: torch.device('cuda:0'))
    device_1: torch.device = field(default_factory=lambda: torch.device('cuda:1'))

    def __post_init__(self) -> None:
        self.test_path = self.data_dir / "test"
        self.test_csv = self.data_dir / "test.csv"
        self.train_path = self.data_dir / "train"
        self.train_csv = self.data_dir / "train.csv"


GLOBAL_CFG = GlobalConfig()


# ====================================================
# Logger
# ====================================================
def init_logger(log_file: str = 'inference.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger(GLOBAL_CFG.output_dir + 'inference.log')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=GLOBAL_CFG.seed)


@contextmanager
def timer(name: str):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.1f}s')


# ====================================================
# Image Preprocessing (Shared - executed once)
# ====================================================
def clean_image(img: np.ndarray) -> np.ndarray:
    """Remove orange date stamp from image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return img


def preprocess_and_cache_images(
        test_df: pd.DataFrame,
        test_path: Path,
) -> Dict[str, np.ndarray]:
    """Pre-process all test images and cache in memory."""
    cache = {}
    unique_images = test_df['image_path'].unique()
    LOGGER.info(f"Pre-processing {len(unique_images)} images...")

    for image_path in tqdm(unique_images, desc="Caching images"):
        file_path = test_path / Path(image_path).name
        img = cv2.imread(str(file_path))
        if img is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        cache[image_path] = img

    LOGGER.info(f"Cached {len(cache)} preprocessed images")
    return cache


# ====================================================
# Dataset (Uses cached images)
# ====================================================
class CachedThreeStreamDataset(Dataset):
    """Dataset that loads from cached images, splits into left/right/full."""

    def __init__(
            self,
            df: pd.DataFrame,
            image_cache: Dict[str, np.ndarray],
            transform: Callable = None,
    ):
        self.file_names = df['image_path'].unique()
        self.image_cache = image_cache
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = self.image_cache[file_name]

        # Split into left/right halves
        height, width = image.shape[:2]
        mid = width // 2
        left = image[:, :mid]
        right = image[:, mid:]
        full_image = image.copy()

        if self.transform:
            transform_left = self.transform()
            transform_right = self.transform()
            transform_full = self.transform()
            left = transform_left(image=left)['image']
            right = transform_right(image=right)['image']
            full_image = transform_full(image=full_image)['image']

        return left, right, full_image, file_name


class CachedThreeStreamDatasetWithMeta(Dataset):
    """Dataset with meta features for models that require them."""

    def __init__(
            self,
            df: pd.DataFrame,
            image_cache: Dict[str, np.ndarray],
            transform: Callable = None,
            meta_df: pd.DataFrame = None,
            default_state: int = 0,
            default_species: int = 0,
            default_ndvi: float = 0.5,
            default_height_norm: float = 0.0,
    ):
        self.file_names = df['image_path'].unique()
        self.image_cache = image_cache
        self.transform = transform
        self.meta_df = meta_df
        self.default_state = default_state
        self.default_species = default_species
        self.default_ndvi = default_ndvi
        self.default_height_norm = default_height_norm

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = self.image_cache[file_name]

        height, width = image.shape[:2]
        mid = width // 2
        left = image[:, :mid]
        right = image[:, mid:]
        full_image = image.copy()

        if self.transform:
            transform_left = self.transform()
            transform_right = self.transform()
            transform_full = self.transform()
            left = transform_left(image=left)['image']
            right = transform_right(image=right)['image']
            full_image = transform_full(image=full_image)['image']

        # Get meta features
        if self.meta_df is not None and file_name in self.meta_df.index:
            row = self.meta_df.loc[file_name]
            state = int(row.get('State_encoded', self.default_state))
            species = int(row.get('Species_encoded', self.default_species))
            ndvi = float(row.get('Pre_GSHH_NDVI', self.default_ndvi))
            height_norm = float(row.get('Height_normalized', self.default_height_norm))
        else:
            state = self.default_state
            species = self.default_species
            ndvi = self.default_ndvi
            height_norm = self.default_height_norm

        state = torch.tensor(state, dtype=torch.long)
        species = torch.tensor(species, dtype=torch.long)
        meta_num = torch.tensor([ndvi, height_norm], dtype=torch.float32)

        return left, right, full_image, state, species, meta_num, file_name


# ====================================================
# Transforms
# ====================================================

def get_tta_exp142_transforms(img_size: int = 1024):
    def base():
        return [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

    return [
        lambda: A.Compose(base()),
        lambda: A.Compose([A.HorizontalFlip(p=1.0), *base()]),
        lambda: A.Compose([A.VerticalFlip(p=1.0), *base()]),  # vertical flip
    ]


def get_tta_transforms(img_size: int = 1024):
    def base():
        return [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

    return [
        lambda: A.Compose(base()),
        lambda: A.Compose([A.HorizontalFlip(p=1.0), *base()]),
        lambda: A.Compose([A.VerticalFlip(p=1.0), *base()]),  # vertical flip
    ]


# ====================================================
# Model Definitions
# ====================================================

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


# ---- Model Type A: Basic 3-stream (exp-v147, exp-v174, exp-v164) ----
class BiomassModelBasic(nn.Module):
    """Basic 3-stream model with optional species head."""

    def __init__(
            self,
            model_name: str = 'vit_large_patch16_dinov3_qkvb',
            pretrained: bool = False,
            img_size: int = 1024,
            num_species: int = 15,
            moe_experts: int = 4,
            moe_hidden: int = 512,
            moe_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            global_pool="avg", img_size=img_size
        )
        self.n_features = self.backbone.num_features * 3

        self.head_total = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_gdm = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_green = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_ndvi = _AuxHead(self.n_features, self.n_features // 2)
        self.head_height = _AuxHead(self.n_features, self.n_features // 2)

        self.head_species = None
        if num_species > 0:
            self.head_species = nn.Sequential(
                nn.Linear(self.n_features, self.n_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.n_features // 2, num_species),
            )

    def forward(self, left, right, full_image):
        feat_left = self.backbone(left)
        feat_right = self.backbone(right)
        feat_full = self.backbone(full_image)
        feats = torch.cat([feat_left, feat_right, feat_full], dim=1)

        total_mean, total_var = self.head_total(feats)
        gdm_mean, gdm_var = self.head_gdm(feats)
        green_mean, green_var = self.head_green(feats)

        return (total_mean, total_var), (gdm_mean, gdm_var), (green_mean, green_var)


# ---- Model Type B: With state/species heads (exp-v142) ----
class BiomassModelWithStateSpecies(nn.Module):
    """Model with State/Species classification heads."""

    def __init__(
            self,
            model_name: str = 'vit_large_patch16_dinov3_qkvb',
            pretrained: bool = False,
            img_size: int = 1024,
            num_species: int = 15,
            num_states: int = 4,
            moe_experts: int = 4,
            moe_hidden: int = 512,
            moe_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            global_pool="avg", img_size=img_size
        )
        self.n_features = self.backbone.num_features * 3

        self.head_total = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_gdm = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_green = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_ndvi = _AuxHead(self.n_features, self.n_features // 2)
        self.head_height = _AuxHead(self.n_features, self.n_features // 2)

        # Auxiliary heads for ratio predictions
        self.head_soil_ratio = _AuxHead(self.n_features, self.n_features // 2)
        self.head_healthy_ratio = _AuxHead(self.n_features, self.n_features // 2)
        self.head_chlorosis_ratio = _AuxHead(self.n_features, self.n_features // 2)

        # Classification heads
        self.head_species = nn.Sequential(
            nn.Linear(self.n_features, self.n_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.n_features // 2, num_species),
        ) if num_species > 0 else None

        self.head_state = nn.Sequential(
            nn.Linear(self.n_features, self.n_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.n_features // 2, num_states),
        ) if num_states > 0 else None

    def forward(self, left, right, full_image):
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


# ---- Model Type C: With meta feature embeddings (exp-v148) ----
class BiomassModelWithMetaEmbedding(nn.Module):
    """Model with meta feature embeddings for State/Species."""

    def __init__(
            self,
            model_name: str = 'vit_large_patch16_dinov3_qkvb',
            pretrained: bool = False,
            img_size: int = 1024,
            num_states: int = 4,
            num_species: int = 15,
            state_embed_dim: int = 8,
            species_embed_dim: int = 16,
            moe_experts: int = 4,
            moe_hidden: int = 512,
            moe_dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            global_pool="avg", img_size=img_size
        )

        n_features_img = self.backbone.num_features * 3

        self.state_embedding = nn.Embedding(num_states, state_embed_dim)
        self.species_embedding = nn.Embedding(num_species, species_embed_dim)

        self.n_features = n_features_img + state_embed_dim + species_embed_dim + 2

        self.head_total = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_gdm = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)
        self.head_green = MetaMoE(self.n_features, moe_hidden, moe_experts, moe_dropout)

    def forward(self, left, right, full_image, state, species, meta_num):
        feat_left = self.backbone(left)
        feat_right = self.backbone(right)
        feat_full = self.backbone(full_image)
        feats_img = torch.cat([feat_left, feat_right, feat_full], dim=1)

        state_emb = self.state_embedding(state)
        species_emb = self.species_embedding(species)

        feats = torch.cat([feats_img, state_emb, species_emb, meta_num], dim=1)

        total_mean, total_var = self.head_total(feats)
        gdm_mean, gdm_var = self.head_gdm(feats)
        green_mean, green_var = self.head_green(feats)

        return (total_mean, total_var), (gdm_mean, gdm_var), (green_mean, green_var)


# ====================================================
# Dual GPU Inference
# ====================================================
@torch.no_grad()
def inference_single_gpu(
        model: nn.Module,
        model_states: List[Dict],
        dataloader: DataLoader,
        device: torch.device,
        has_meta: bool = False,
) -> Dict[str, np.ndarray]:
    """Run inference on single GPU. Handles empty dataloader gracefully."""
    model.to(device)

    all_total, all_gdm, all_green = [], [], []
    all_ndvi, all_height = [], []
    all_file_names = []
    all_state_logits, all_species_logits = [], []

    for batch in dataloader:
        if has_meta:
            left, right, full_img, state, species, meta_num, file_names = batch
            left, right, full_img = left.to(device), right.to(device), full_img.to(device)
            state, species = state.to(device), species.to(device)
            meta_num = meta_num.to(device)
        else:
            left, right, full_img, file_names = batch
            left, right, full_img = left.to(device), right.to(device), full_img.to(device)

        batch_total, batch_gdm, batch_green = [], [], []
        batch_ndvi, batch_height = [], []
        batch_state_logits, batch_species_logits = [], []

        for state_dict in model_states:
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            with torch.no_grad():
                if has_meta:
                    preds = model(left, right, full_img, state, species, meta_num)
                else:
                    preds = model(left, right, full_img)

            # Extract predictions
            total_mean = preds[0][0].cpu().numpy()
            gdm_mean = preds[1][0].cpu().numpy()
            green_mean = preds[2][0].cpu().numpy()

            batch_total.append(total_mean)
            batch_gdm.append(gdm_mean)
            batch_green.append(green_mean)

            if len(preds) >= 10:
                ndvi_mean = preds[3][0].cpu().numpy()
                height_mean = preds[4][0].cpu().numpy()
                batch_ndvi.append(ndvi_mean)
                batch_height.append(height_mean)
                # Species/State logits at positions 8 and 9
                if preds[8] is not None:
                    batch_species_logits.append(preds[8].cpu().numpy())
                if preds[9] is not None:
                    batch_state_logits.append(preds[9].cpu().numpy())
            else:
                # BiomassModelBasic or BiomassModelWithMetaEmbedding (3 outputs)
                # No NDVI/Height predictions, no species/state logits
                pass

        all_total.append(np.mean(batch_total, axis=0).flatten())
        all_gdm.append(np.mean(batch_gdm, axis=0).flatten())
        all_green.append(np.mean(batch_green, axis=0).flatten())
        all_file_names.extend(file_names)

        if batch_ndvi:
            all_ndvi.append(np.mean(batch_ndvi, axis=0).flatten())
        if batch_height:
            all_height.append(np.mean(batch_height, axis=0).flatten())
        if batch_state_logits:
            all_state_logits.append(np.mean(batch_state_logits, axis=0))
        if batch_species_logits:
            all_species_logits.append(np.mean(batch_species_logits, axis=0))

    # Handle empty dataloader case
    if len(all_total) == 0:
        return {
            'total': np.array([]),
            'gdm': np.array([]),
            'green': np.array([]),
            'file_names': [],
        }

    result = {
        'total': np.concatenate(all_total),
        'gdm': np.concatenate(all_gdm),
        'green': np.concatenate(all_green),
        'file_names': all_file_names,
    }

    if all_ndvi:
        result['ndvi'] = np.concatenate(all_ndvi)
    if all_height:
        result['height'] = np.concatenate(all_height)
    if all_state_logits:
        result['state_logits'] = np.vstack(all_state_logits)
    if all_species_logits:
        result['species_logits'] = np.vstack(all_species_logits)

    return result


def parallel_dual_gpu_inference(
        dataset: Dataset,
        model_class: type,
        model_kwargs: Dict,
        model_states: List[Dict],
        batch_size: int = 2,
        num_workers: int = 0,
        has_meta: bool = False,
) -> Dict[str, np.ndarray]:
    """Run inference in parallel on two GPUs.

    Falls back to single GPU if dataset has only 1 sample.
    """
    device_0 = GLOBAL_CFG.device_0
    device_1 = GLOBAL_CFG.device_1

    n = len(dataset)

    # Edge case: only 1 sample, can't split -> use single GPU
    if n <= 1:
        LOGGER.info(f"Dataset has {n} sample(s), using single GPU inference")
        model = model_class(**model_kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        start = time.time()
        result = inference_single_gpu(model, model_states, loader, device_0, has_meta)
        LOGGER.info(f"Single GPU inference completed in {time.time() - start:.1f}s")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return result

    # Split dataset for dual GPU
    indices_0 = list(range(0, n, 2))  # Even indices -> GPU 0
    indices_1 = list(range(1, n, 2))  # Odd indices -> GPU 1
    dataset_0 = Subset(dataset, indices_0)
    dataset_1 = Subset(dataset, indices_1)

    LOGGER.info(f"GPU0: {len(dataset_0)} samples, GPU1: {len(dataset_1)} samples")

    # Create models
    model_0 = model_class(**model_kwargs)
    model_1 = model_class(**model_kwargs)

    # Dataloaders
    loader_0 = DataLoader(dataset_0, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    loader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    start = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_0 = executor.submit(inference_single_gpu, model_0, model_states,
                                   loader_0, device_0, has_meta)
        future_1 = executor.submit(inference_single_gpu, model_1, model_states,
                                   loader_1, device_1, has_meta)
        results_0 = future_0.result()
        results_1 = future_1.result()

    LOGGER.info(f"Parallel inference completed in {time.time() - start:.1f}s")

    # Merge results
    total = np.zeros(n)
    gdm = np.zeros(n)
    green = np.zeros(n)
    file_names = [''] * n

    # Merge GPU 0 results
    for i, idx in enumerate(indices_0):
        total[idx] = results_0['total'][i]
        gdm[idx] = results_0['gdm'][i]
        green[idx] = results_0['green'][i]
        file_names[idx] = results_0['file_names'][i]

    # Merge GPU 1 results (handle empty case)
    if len(results_1['total']) > 0:
        for i, idx in enumerate(indices_1):
            total[idx] = results_1['total'][i]
            gdm[idx] = results_1['gdm'][i]
            green[idx] = results_1['green'][i]
            file_names[idx] = results_1['file_names'][i]

    result = {
        'total': total,
        'gdm': gdm,
        'green': green,
        'file_names': file_names,
    }

    # Merge state/species logits if present in both results
    if 'state_logits' in results_0 and results_0['state_logits'].size > 0:
        num_classes = results_0['state_logits'].shape[1]
        state_logits = np.zeros((n, num_classes))
        for i, idx in enumerate(indices_0):
            state_logits[idx] = results_0['state_logits'][i]
        if 'state_logits' in results_1 and results_1['state_logits'].size > 0:
            for i, idx in enumerate(indices_1):
                state_logits[idx] = results_1['state_logits'][i]
        result['state_logits'] = state_logits

    if 'species_logits' in results_0 and results_0['species_logits'].size > 0:
        num_classes = results_0['species_logits'].shape[1]
        species_logits = np.zeros((n, num_classes))
        for i, idx in enumerate(indices_0):
            species_logits[idx] = results_0['species_logits'][i]
        if 'species_logits' in results_1 and results_1['species_logits'].size > 0:
            for i, idx in enumerate(indices_1):
                species_logits[idx] = results_1['species_logits'][i]
        result['species_logits'] = species_logits

    # Merge NDVI predictions if present
    if 'ndvi' in results_0 and len(results_0['ndvi']) > 0:
        ndvi = np.zeros(n)
        for i, idx in enumerate(indices_0):
            ndvi[idx] = results_0['ndvi'][i]
        if 'ndvi' in results_1 and len(results_1['ndvi']) > 0:
            for i, idx in enumerate(indices_1):
                ndvi[idx] = results_1['ndvi'][i]
        result['ndvi'] = ndvi

    # Merge Height predictions if present
    if 'height' in results_0 and len(results_0['height']) > 0:
        height = np.zeros(n)
        for i, idx in enumerate(indices_0):
            height[idx] = results_0['height'][i]
        if 'height' in results_1 and len(results_1['height']) > 0:
            for i, idx in enumerate(indices_1):
                height[idx] = results_1['height'][i]
        result['height'] = height

    # Cleanup
    del model_0, model_1
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ====================================================
# Post Processing
# ====================================================
def apply_post_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Clover and Dead post-processing rules."""
    df_out = df.copy()
    if 'Dry_Clover_g' in df_out.columns:
        df_out['Dry_Clover_g'] = df_out['Dry_Clover_g'] * 0.8
    if 'Dry_Dead_g' in df_out.columns:
        mask_high = df_out['Dry_Dead_g'] > 20
        df_out.loc[mask_high, 'Dry_Dead_g'] *= 1.1
        mask_low = df_out['Dry_Dead_g'] < 10
        df_out.loc[mask_low, 'Dry_Dead_g'] *= 0.9
    return df_out


# ====================================================
# Model Configurations
# ====================================================
MODEL_CONFIGS = [
    {
        "name": "exp-v142",
        "model_dir": "/kaggle/input/exp1394243-csiro-fullheadaux/pytorch/default/1/exp-v142",
        "model_class": BiomassModelWithStateSpecies,
        "model_kwargs": {
            "model_name": "vit_large_patch16_dinov3_qkvb",
            "pretrained": False,
            "img_size": 1024,
            "num_species": 15,
            "num_states": 4,
        },
        "has_meta": False,
        "extract_state": True,
        "weight": 0,
        "folds": [0],
    },
    {
        "name": "exp-v148",
        "model_dir": "/kaggle/input/exp148177-csiro-4aug/pytorch/default/1/exp-v148",
        "model_class": BiomassModelWithMetaEmbedding,
        "model_kwargs": {
            "model_name": "vit_large_patch16_dinov3_qkvb",
            "pretrained": False,
            "img_size": 1024,
            "num_states": 4,
            "num_species": 15,
            "state_embed_dim": 8,
            "species_embed_dim": 16,
        },
        "has_meta": True,
        "extract_state": False,
        "weight": 0.15,
        "folds": [0],
    },
    {
        "name": "exp-v147",
        "model_dir": "/kaggle/input/exp147-csiro-136seed/pytorch/default/1",
        "model_class": BiomassModelBasic,
        "model_kwargs": {
            "model_name": "vit_large_patch16_dinov3_qkvb",
            "pretrained": False,
            "img_size": 1024,
            "num_species": 15
        },
        "has_meta": False,
        "extract_state": False,
        "weight": 0.2,
        "folds": [0],
    },

    {
        "name": "exp-v174",
        "model_dir": "/kaggle/input/exp1745678-csiro-aug/pytorch/default/1/exp-v174",
        "model_class": BiomassModelBasic,
        "model_kwargs": {
            "model_name": "vit_large_patch16_dinov3_qkvb",
            "pretrained": False,
            "img_size": 1024,
            "num_species": 15,
        },
        "has_meta": False,
        "extract_state": False,
        "weight": 0.25,
        "folds": [0],
    },
    {
        "name": "exp-v164",
        "model_dir": "/kaggle/input/exp164-cisro-1474aug/pytorch/default/2",
        "model_class": BiomassModelBasic,
        "model_kwargs": {
            "model_name": "vit_large_patch16_dinov3_qkvb",
            "pretrained": False,
            "img_size": 1024,
            "num_species": 15,
        },
        "has_meta": False,
        "extract_state": False,
        "weight": 0.2,
        "folds": [0],
    },
    {
        "name": "exp-v177",
        "model_dir": "/kaggle/input/exp1745678-csiro-aug/pytorch/default/1/exp-v177",
        "model_class": BiomassModelBasic,
        "model_kwargs": {
            "model_name": "vit_large_patch16_dinov3_qkvb",
            "pretrained": False,
            "img_size": 1024,
            "num_species": 15,
        },
        "has_meta": False,
        "extract_state": False,
        "weight": 0.2,
        "folds": [0],
    },
]

# ====================================================
# Main Inference Pipeline
# ====================================================
LOGGER.info("=" * 60)
LOGGER.info("CSIRO Ensemble V10 - Dual GPU Inference")
LOGGER.info("=" * 60)

# Load test data
test = pd.read_csv(GLOBAL_CFG.test_csv)

# if len(test)<10:
#     test = pd.read_csv(GLOBAL_CFG.train_csv)[['sample_id','image_path','target_name']]
#     GLOBAL_CFG.test_path=GLOBAL_CFG.train_path

LOGGER.info(f"Loaded {len(test)} test samples")
LOGGER.info(f"Unique images: {test['image_path'].nunique()}")

# Pre-process and cache all images (ONCE)
with timer("Image Preprocessing"):
    image_cache = preprocess_and_cache_images(test, GLOBAL_CFG.test_path)

# Get TTA transforms

# Store model predictions
all_model_preds = []
predicted_states_df = None
meta_df = None

# Run inference for each model
for config in MODEL_CONFIGS:

    LOGGER.info("=" * 40)
    LOGGER.info(f"Model: {config['name']}")
    LOGGER.info("=" * 40)
    if config['name'] != 'exp-v142':
        tta_transforms = get_tta_transforms(GLOBAL_CFG.img_size)
    else:
        tta_transforms = get_tta_exp142_transforms(GLOBAL_CFG.img_size)

    LOGGER.info(f"tta nums:{len(tta_transforms)}")
    # Load model states
    model_states = []
    for fold in config['folds']:
        model_path = Path(config['model_dir']) / f'fold{fold}_best.pth'
        if model_path.exists():
            state = torch.load(model_path, map_location='cpu')
            model_states.append(state)
            LOGGER.info(f"Loaded fold {fold}")
        else:
            LOGGER.warning(f"Not found: {model_path}")

    if not model_states:
        LOGGER.warning(f"Skipping {config['name']} - no model files found")
        continue

    # TTA views
    tta_preds = []
    for view_idx, transform in enumerate(tta_transforms, start=1):
        LOGGER.info(f"TTA view {view_idx}/{len(tta_transforms)}")

        # Create dataset based on model requirements
        if config['has_meta']:
            # Need meta features for this model
            if meta_df is None:
                # Load meta from exp-v142 predictions or use defaults
                meta_csv = Path('/kaggle/working/predicted_meta.csv')
                if meta_csv.exists():
                    meta = pd.read_csv(meta_csv)
                    STATE_CLASSES = ['NSW', 'Tas', 'Vic', 'WA']
                    SPECIES_CLASSES = [
                        'Clover', 'Fescue', 'Fescue_CrumbWeed', 'Lucerne', 'Mixed',
                        'Phalaris', 'Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed',
                        'Phalaris_Clover', 'Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass',
                        'Phalaris_Ryegrass_Clover', 'Ryegrass', 'Ryegrass_Clover',
                        'SubcloverDalkeith', 'SubcloverLosa', 'WhiteClover'
                    ]
                    state_to_idx = {s: i for i, s in enumerate(STATE_CLASSES)}
                    species_to_idx = {s: i for i, s in enumerate(SPECIES_CLASSES)}
                    meta['State_encoded'] = meta['State'].map(state_to_idx).fillna(0).astype(int)
                    meta['Species_encoded'] = meta['Species'].map(species_to_idx).fillna(0).astype(int)
                    meta['Height_normalized'] = (meta['Height_Ave_cm'] - 20.0) / 10.0
                    meta_df = meta.set_index('image_path')
                    LOGGER.info(f"Loaded meta features from {meta_csv}")
                else:
                    LOGGER.warning("Meta CSV not found, using defaults")

            dataset = CachedThreeStreamDatasetWithMeta(
                test, image_cache, transform=transform, meta_df=meta_df
            )
        else:
            dataset = CachedThreeStreamDataset(test, image_cache, transform=transform)

        with timer(f"TTA {view_idx}"):
            results = parallel_dual_gpu_inference(
                dataset,
                config['model_class'],
                config['model_kwargs'],
                model_states,
                batch_size=GLOBAL_CFG.batch_size,
                num_workers=GLOBAL_CFG.num_workers,
                has_meta=config['has_meta'],
            )

        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'image_path': results['file_names'],
            'Dry_Total_g': results['total'],
            'GDM_g': results['gdm'],
            'Dry_Green_g': results['green'],
            'Dry_Dead_g': np.maximum(0, results['total'] - results['gdm']),
            'Dry_Clover_g': np.maximum(0, results['gdm'] - results['green']),
        })
        pred_df = pred_df.sort_values('image_path').reset_index(drop=True)
        tta_preds.append(pred_df)

        # Extract state predictions from first model
        if config['extract_state'] and 'state_logits' in results and predicted_states_df is None:
            state_idx = np.argmax(results['state_logits'], axis=1)
            STATE_CLASSES = ['NSW', 'Tas', 'Vic', 'WA']
            state_names = [STATE_CLASSES[i] for i in state_idx]
            predicted_states_df = pd.DataFrame({
                'image_path': results['file_names'],
                'Predicted_State': state_names,
            })
            predicted_states_df.to_csv('predicted_states.csv', index=False)

            # Also save meta for exp-v148
            if 'species_logits' in results:
                SPECIES_CLASSES = [
                    'Clover', 'Fescue', 'Fescue_CrumbWeed', 'Lucerne', 'Mixed',
                    'Phalaris', 'Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed',
                    'Phalaris_Clover', 'Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass',
                    'Phalaris_Ryegrass_Clover', 'Ryegrass', 'Ryegrass_Clover',
                    'SubcloverDalkeith', 'SubcloverLosa', 'WhiteClover'
                ]
                species_idx = np.argmax(results['species_logits'], axis=1)
                species_names = [SPECIES_CLASSES[i] for i in species_idx]
                # Use predicted NDVI and Height values if available
                pred_ndvi = results.get('ndvi', np.full(len(results['file_names']), 0.5))
                pred_height = results.get('height', np.full(len(results['file_names']), 20.0))
                meta_out = pd.DataFrame({
                    'image_path': results['file_names'],
                    'State': state_names,
                    'Species': species_names,
                    'Pre_GSHH_NDVI': pred_ndvi,
                    'Height_Ave_cm': pred_height,
                })
                meta_out.to_csv('predicted_meta.csv', index=False)
                LOGGER.info("Saved predicted_meta.csv")
    del tta_transforms
    # Average TTA predictions
    if len(tta_preds) > 1:
        final_pred = tta_preds[0].copy()
        value_cols = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
        for col in value_cols:
            stacked = np.stack([df[col].values for df in tta_preds], axis=0)
            final_pred[col] = stacked.mean(axis=0)
    else:
        final_pred = tta_preds[0]

    # Apply WA post-processing
    if predicted_states_df is not None:
        final_pred = final_pred.merge(predicted_states_df, on='image_path', how='left')
        is_wa = final_pred['Predicted_State'] == 'WA'
        final_pred.loc[is_wa, 'Dry_Dead_g'] = 0
        final_pred.loc[is_wa, 'Dry_Total_g'] = final_pred.loc[is_wa, 'GDM_g']

    # Apply general post-processing
    if config['name'] != 'exp-v142':
        final_pred = apply_post_processing(final_pred)
    LOGGER.info(final_pred.head())
    all_model_preds.append({
        'name': config['name'],
        'pred_df': final_pred,
        'weight': config['weight'],
    })

    LOGGER.info(f"{config['name']} predictions completed")

# ====================================================
# Ensemble
# ====================================================
LOGGER.info("=" * 40)
LOGGER.info("Ensembling predictions")
LOGGER.info("=" * 40)

# Weighted ensemble
weights = [m['weight'] for m in all_model_preds]
pred_dfs = [m['pred_df'] for m in all_model_preds]

# Ensure all have same order
for df in pred_dfs:
    df.sort_values('image_path', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Weighted average
final_ensemble = pred_dfs[0][['image_path']].copy()
value_cols = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
for col in value_cols:
    stacked = np.stack([df[col].values for df in pred_dfs], axis=0)
    weights_arr = np.array(weights).reshape(-1, 1)
    final_ensemble[col] = (stacked * weights_arr).sum(axis=0)

LOGGER.info(f"Ensemble weights: {dict(zip([m['name'] for m in all_model_preds], weights))}")

# ====================================================
# Create Submission
# ====================================================
test_with_preds = test.merge(final_ensemble, on='image_path', how='left')


def get_target_value(row):
    target_name = row['target_name']
    return row[target_name]


test_with_preds['target'] = test_with_preds.apply(get_target_value, axis=1)

test_with_preds['target'] = test_with_preds['target'].round(4)  # 保留后四位

submission = test_with_preds[['sample_id', 'target']].copy()
submission.to_csv('submission.csv', index=False)

LOGGER.info(f"Submission saved: {submission.shape}")
LOGGER.info("=" * 60)
LOGGER.info("Inference Complete!")
LOGGER.info("=" * 60)

submission.head()

# %% [code] {"jupyter":{"outputs_hidden":false}}
