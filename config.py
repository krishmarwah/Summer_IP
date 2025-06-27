import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"

# -----------------------------------------------------------------------------
# Core Training Hyper-parameters
# -----------------------------------------------------------------------------
BATCH_SIZE      = 5     # 4–8 on a 3090; original CycleGAN used 1
NUM_EPOCHS      = 500         # 200 epochs with linear LR decay is standard
LEARNING_RATE   = 2e-4        # Start at 2e-4, then linearly decay after 100 epochs
LAMBDA_CYCLE    = 10          # Enforce cycle consistency
LAMBDA_IDENTITY = 5           # Helps preserve color/style when mapping same-domain
NUM_WORKERS     = 4           # DataLoader workers
LR_DECAY_START_EPOCH = 20  # or whatever epoch you want to begin the decay

# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
LOAD_MODEL        = True
SAVE_MODEL        = True
CHECKPOINT_GEN_H  = "genh.pth.tar"
CHECKPOINT_GEN_Z  = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

# -----------------------------------------------------------------------------
# Projection-Loss (Manifold-aware PCA penalty)
# -----------------------------------------------------------------------------
USE_PROJECTION_LOSS    = True
PROJECTION_LOSS_K      = 3     # k-nearest neighbors in feature/PCA space
PROJECTION_LOSS_R      = 2     # top-r principal components
PROJECTION_LOSS_WEIGHT = 0.1   # small weight so it doesn’t overwhelm GAN+cycle

# -----------------------------------------------------------------------------
# Data Augmentations (256×256 pipeline)
# -----------------------------------------------------------------------------
transforms = A.Compose([
    A.Resize(286, 286),                # pad / scale up slightly
    A.RandomCrop(256,256),            # crop back to 256
    A.HorizontalFlip(p=0.5),           # random mirror
    A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255),
    ToTensorV2(),
], additional_targets={"image0": "image"})

# -----------------------------------------------------------------------------
# FID / IS Evaluation settings
# -----------------------------------------------------------------------------
RUN_FID_EVALUATION = True
NUM_FID_IMAGES     = 1000          # generate 1k per direction
FID_BATCH_SIZE     = 16            # batch for generation
FID_SAVE_DIR       = "saved_images"

fid_transform = A.Compose([
    A.Resize(128, 128),              # FID networks typically expect 128 or 299
    A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255),
    ToTensorV2(),
])
