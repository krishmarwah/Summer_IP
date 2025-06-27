"""
Training for CycleGAN + Projection Loss
"""
import os
import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.linalg
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import HorseZebraDataset
from discriminator_model import Discriminator
from generator_model import Generator
from utils import save_checkpoint, load_checkpoint
import config

# torchmetrics imports if you want FID/IS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms

"""# Load ResNet-50 once globally and freeze it
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])  # remove avgpool and FC
resnet50.eval().to("cuda")  # or use config.DEVICE
for param in resnet50.parameters():
    param.requires_grad = False

def extract_resnet_features(x):
    Extract ResNet-50 convolutional features from input images.
    with torch.no_grad():
        feats = resnet50(x)  # Shape: (B, 2048, H', W')
        feats = F.adaptive_avg_pool2d(feats, (8, 8))  # Control spatial size
        return feats.view(x.shape[0], -1)  # Shape: (B, 2048 * 8 * 8)
"""

# For tracking losses
generator_losses = []
projection_losses = []

def projection_loss(x_real, x_fake, k=3, r=2, eps=1e-6):
    B, C, H, W = x_real.shape
    # Handle small batch sizes gracefully
    if B < 2:
        return torch.tensor(0.0, device=x_real.device)
    
    k_eff = min(k, B - 1)
    r_eff = min(r, k_eff)
    
    # Downsample to reduce memory usage
    x_real_small = F.interpolate(x_real, scale_factor=0.25, mode='bilinear', align_corners=False)
    x_fake_small = F.interpolate(x_fake, scale_factor=0.25, mode='bilinear', align_corners=False)
    
    # Convert to float32 for SVD operations
    x_real_small = x_real_small.float()
    x_fake_small = x_fake_small.float()
    
    x_real_flat = x_real_small.view(B, -1)
    x_fake_flat = x_fake_small.view(B, -1)

    with torch.no_grad():
        # Efficient distance computation
        dist = torch.cdist(x_real_flat, x_real_flat, p=2)
        _, nn_idx = torch.topk(dist, k_eff + 1, largest=False, dim=-1)
        nn_idx = nn_idx[:, 1:]

    if r == 0:
        real_mean = x_real_flat.mean(dim=0)
        fake_mean = x_fake_flat.mean(dim=0)
        return torch.norm(real_mean - fake_mean, p=2)

    losses = []
    for i in range(B):
        idx = nn_idx[i]
        Xi = x_real_flat[idx]
        mu = Xi.mean(dim=0, keepdim=True)
        Xi_cent = Xi - mu
        
        # Full SVD with float32 - slower but compatible
        U, _, _ = torch.linalg.svd(Xi_cent.t(), full_matrices=False)
        U_r = U[:, :r_eff]
        P_real = U_r @ U_r.t()

        Xif = x_fake_flat[idx]
        muf = Xif.mean(dim=0, keepdim=True)
        Xif_cent = Xif - muf
        
        V, _, _ = torch.linalg.svd(Xif_cent.t(), full_matrices=False)
        V_r = V[:, :r_eff]
        P_fake = V_r @ V_r.t()

        # Efficient matrix difference computation
        diff = P_real - P_fake
        losses.append(torch.sum(diff * diff))  # Frobenius norm squared

    # Convert back to original dtype
    loss_tensor = torch.stack(losses).mean()
    return loss_tensor.to(x_real.dtype)

"""def projection_loss(x_real, x_fake, k=3, r=2, eps=1e-6):
    Projection loss using pretrained ResNet-50 features.
    Args:
        x_real, x_fake: (B, C, H, W)
        k: # neighbors for local PCA
        r: # principal components
    Returns:
        Scalar projection loss (Frobenius norm between projection matrices)
    B = x_real.size(0)
    if B < 2:
        return torch.tensor(0.0, device=x_real.device)

    # Extract frozen features using pretrained ResNet (always float32)
    x_real_feat = extract_resnet_features(x_real).float()  # (B, D)
    x_fake_feat = extract_resnet_features(x_fake).float()  # (B, D)

    k_eff = min(k, B - 1)
    r_eff = min(r, k_eff)

    with torch.no_grad():
        dist_real = torch.cdist(x_real_feat, x_real_feat)
        _, nn_idx_real = torch.topk(dist_real, k_eff + 1, largest=False, dim=-1)
        nn_idx_real = nn_idx_real[:, 1:]

        dist_fake = torch.cdist(x_fake_feat, x_fake_feat)
        _, nn_idx_fake = torch.topk(dist_fake, k_eff + 1, largest=False, dim=-1)
        nn_idx_fake = nn_idx_fake[:, 1:]

    losses = []
    for i in range(B):
        Xi = x_real_feat[nn_idx_real[i]].float()
        Xi_cent = Xi - Xi.mean(dim=0, keepdim=True)
        U, _, _ = torch.linalg.svd(Xi_cent.T, full_matrices=False)
        P_real = U[:, :r_eff] @ U[:, :r_eff].T

        Xf = x_fake_feat[nn_idx_fake[i]].float()
        Xf_cent = Xf - Xf.mean(dim=0, keepdim=True)
        V, _, _ = torch.linalg.svd(Xf_cent.T, full_matrices=False)
        P_fake = V[:, :r_eff] @ V[:, :r_eff].T

        losses.append(torch.sum((P_real - P_fake) ** 2))

    return torch.stack(losses).mean().to(x_real.dtype)
"""

""" Remove Dwonsampling
    Use pretrained resnet 50 model use model.evolve(nearest neighbors feed it to svd)
    resnet 50 isnt supposed to be trained pretrained
    use in evolve mode to make sure nothing chages in 
    resnet 50 ->features and not probabilities
    obtain featurs from that 
    use those features to find the nearest neighbors
"""
"""def projection_loss(x_real, x_fake, k=3, r=2, eps=1e-6):
    
    Differentiable Projected PCA Loss for Manifold-Aware Learning.

    Args:
        x_real (Tensor): Real images, shape (B, C, H, W)
        x_fake (Tensor): Generated images, shape (B, C, H, W)
        k (int): Number of nearest neighbors
        r (int): Number of principal components for projection
        eps (float): Small number for numerical stability (currently unused)

    Returns:
        torch.Tensor: Scalar projection loss
    B, C, H, W = x_real.shape
    if B < 2:
        return torch.tensor(0.0, device=x_real.device)

    # Downsample to reduce memory and flatten
    x_real_ds = F.interpolate(x_real, scale_factor=0.25, mode='bilinear', align_corners=False).float()
    x_fake_ds = F.interpolate(x_fake, scale_factor=0.25, mode='bilinear', align_corners=False).float()

    x_real_flat = x_real_ds.view(B, -1)
    x_fake_flat = x_fake_ds.view(B, -1)

    k_eff = min(k, B - 1)
    r_eff = min(r, k_eff)

    with torch.no_grad():
        # Compute separate k-NNs for real and fake
        dist_real = torch.cdist(x_real_flat, x_real_flat, p=2)
        _, nn_idx_real = torch.topk(dist_real, k_eff + 1, largest=False, dim=-1)
        nn_idx_real = nn_idx_real[:, 1:]

        dist_fake = torch.cdist(x_fake_flat, x_fake_flat, p=2)
        _, nn_idx_fake = torch.topk(dist_fake, k_eff + 1, largest=False, dim=-1)
        nn_idx_fake = nn_idx_fake[:, 1:]

    losses = []
    for i in range(B):
        # Neighborhood for real sample
        Xi = x_real_flat[nn_idx_real[i]]
        mu_i = Xi.mean(dim=0, keepdim=True)
        Xi_centered = Xi - mu_i
        U, _, _ = torch.linalg.svd(Xi_centered.t(), full_matrices=False)
        U_r = U[:, :r_eff]
        P_real = U_r @ U_r.t()

        # Neighborhood for fake sample
        Xf = x_fake_flat[nn_idx_fake[i]]
        mu_f = Xf.mean(dim=0, keepdim=True)
        Xf_centered = Xf - mu_f
        V, _, _ = torch.linalg.svd(Xf_centered.t(), full_matrices=False)
        V_r = V[:, :r_eff]
        P_fake = V_r @ V_r.t()

        # Frobenius norm squared
        diff = P_real - P_fake
        loss = torch.sum(diff * diff)
        losses.append(loss)

    return torch.stack(losses).mean().to(x_real.dtype)
"""

def save_triplet(input_img, generated_img, cycle_img, filename):    
    imgs = torch.cat([input_img, generated_img, cycle_img], dim=0)
    grid = make_grid(imgs, nrow=3, normalize=True, value_range=(-1, 1))
    save_image(grid, filename)

def adjust_learning_rate(optimizer, init_lr, epoch):
    if epoch >= config.LR_DECAY_START_EPOCH:
        lr = init_lr * (config.NUM_EPOCHS - epoch) / (config.NUM_EPOCHS - config.LR_DECAY_START_EPOCH)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)
    total_g_loss = 0.0
    total_proj_loss = 0.0

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Discriminator update
        with torch.amp.autocast(device_type="cuda"):
            fake_horse = gen_H(zebra)
            D_H_loss = mse(disc_H(horse), torch.ones_like(disc_H(horse))) + \
                       mse(disc_H(fake_horse.detach()), torch.zeros_like(disc_H(fake_horse)))

            fake_zebra = gen_Z(horse)
            D_Z_loss = mse(disc_Z(zebra), torch.ones_like(disc_Z(zebra))) + \
                       mse(disc_Z(fake_zebra.detach()), torch.zeros_like(disc_Z(fake_zebra)))

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Generator update
        with torch.amp.autocast(device_type="cuda"):
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            G_base = (
                loss_G_Z + loss_G_H +
                cycle_zebra_loss * config.LAMBDA_CYCLE +
                cycle_horse_loss * config.LAMBDA_CYCLE +
                identity_horse_loss * config.LAMBDA_IDENTITY +
                identity_zebra_loss * config.LAMBDA_IDENTITY
            )

            if config.USE_PROJECTION_LOSS:
                proj_z = projection_loss(zebra, fake_zebra, 
                            k=config.PROJECTION_LOSS_K, 
                            r=config.PROJECTION_LOSS_R)
                proj_h = projection_loss(horse, fake_horse, 
                            k=config.PROJECTION_LOSS_K, 
                            r=config.PROJECTION_LOSS_R)
                proj = proj_z + proj_h
                α = G_base.detach() / (proj.detach() + 1e-6)
                α = torch.clamp(α, min=0.1, max=10.0)
                G_loss = G_base + α * proj *config.PROJECTION_LOSS_WEIGHT   # Amplify projection contribution

                if idx == 0:
                    print(f"[Loss Debug] G_base: {G_base.item():.4f}, proj_z: {proj_z.item():.6f}, proj_h: {proj_h.item():.6f}, α: {α.item():.4f}")
            else:
                G_loss = G_base

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        total_g_loss += G_loss.item()
        total_proj_loss += proj.item() if config.USE_PROJECTION_LOSS else 0.0

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=D_H_fake.mean().item(), Z_real=D_Z_fake.mean().item())

    generator_losses.append(total_g_loss / len(loader))
    projection_losses.append(total_proj_loss / len(loader))



def main():
    # Models
    disc_H = Discriminator(3).to(config.DEVICE)
    disc_Z = Discriminator(3).to(config.DEVICE)
    gen_Z = Generator(3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(3, num_residuals=9).to(config.DEVICE)
    # Opts
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()), lr=config.LEARNING_RATE, betas=(0.5,0.999)
    )
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()), lr=config.LEARNING_RATE, betas=(0.5,0.999)
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    # Load
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE)
    # Datasets
    dataset = HorseZebraDataset(
        f"{config.TRAIN_DIR}/horses",
        f"{config.TRAIN_DIR}/zebras",
        transform=config.transforms
    )
    # fixed samples
    fixed_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    fixed_zebra, fixed_horse = next(iter(fixed_loader))
    fixed_zebra, fixed_horse = fixed_zebra.to(config.DEVICE), fixed_horse.to(config.DEVICE)
    # training loader
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        adjust_learning_rate(opt_gen, config.LEARNING_RATE, epoch)
        adjust_learning_rate(opt_disc, config.LEARNING_RATE, epoch)

        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader,
                 opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, config.CHECKPOINT_CRITIC_Z)
        # fixed visuals
        with torch.no_grad():
            fake_horse_fixed = gen_H(fixed_zebra)
            fake_zebra_fixed = gen_Z(fixed_horse)
            cycle_zebra_fixed = gen_Z(fake_horse_fixed)
            cycle_horse_fixed = gen_H(fake_zebra_fixed)
            save_image(fixed_zebra*0.5+0.5, "saved_images/fixed_input_zebra.png")
            save_image(fixed_horse*0.5+0.5, "saved_images/fixed_input_horse.png")
            save_image(fake_horse_fixed*0.5+0.5, f"saved_images/fixed_horse_epoch{epoch+1}.png")
            save_image(fake_zebra_fixed*0.5+0.5, f"saved_images/fixed_zebra_epoch{epoch+1}.png")
            # triplets
            save_triplet(fixed_horse, fake_zebra_fixed, cycle_horse_fixed,
                         f"saved_images/triplet_horse_epoch{epoch+1}.png")
            save_triplet(fixed_zebra, fake_horse_fixed, cycle_zebra_fixed,
                         f"saved_images/triplet_zebra_epoch{epoch+1}.png")

    
    # plot
    plt.figure(figsize=(10,6))
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(projection_losses, label="Projection Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    print("Saved loss_plot.png")



    if config.RUN_FID_EVALUATION:
        # ──────────────────────────────────────────────────────
        # 1) Generate 1k horse→zebra and 1k zebra→horse images
        # ──────────────────────────────────────────────────────
      
        # make sure dirs exist
        os.makedirs("saved_images/horse2zebra", exist_ok=True)
        os.makedirs("saved_images/zebra2horse", exist_ok=True)

        # 1a) horse → zebra
        horse_ds = HorseZebraDataset(
            root_horse=os.path.join(config.TRAIN_DIR, "horses"),
            root_zebra=None,
            transform=config.fid_transform
        )
        loader_h = DataLoader(horse_ds,
                              batch_size=config.FID_BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS)
        cnt = 0
        for _, horses in loader_h:
            horses = horses.to(config.DEVICE)
            with torch.no_grad():
                fake_z = gen_Z(horses)
            for img in fake_z:
                if cnt >= config.NUM_FID_IMAGES: break
                save_image(img * 0.5 + 0.5,
                           f"saved_images/horse2zebra/h2z_{cnt:04d}.jpg")
                cnt += 1
            if cnt >= config.NUM_FID_IMAGES: break

        # 1b) zebra → horse
        zebra_ds = HorseZebraDataset(
            root_horse=None,
            root_zebra=os.path.join(config.TRAIN_DIR, "zebras"),
            transform=config.fid_transform
        )
        loader_z = DataLoader(zebra_ds,
                              batch_size=config.FID_BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS)
        cnt = 0
        for zebras, _ in loader_z:
            zebras = zebras.to(config.DEVICE)
            with torch.no_grad():
                fake_h = gen_H(zebras)
            for img in fake_h:
                if cnt >= config.NUM_FID_IMAGES: break
                save_image(img * 0.5 + 0.5,
                           f"saved_images/zebra2horse/z2h_{cnt:04d}.jpg")
                cnt += 1
            if cnt >= config.NUM_FID_IMAGES: break

        print(f"✅ Generated {config.NUM_FID_IMAGES} fakes each way.")
        # ──────────────────────────────────────────────────────

        # ──────────────────────────────────────────────────────
        # 2) Compute FID (and IS) right here
        # ──────────────────────────────────────────────────────
        
        def all_images(folder):
            return glob.glob(f"{folder}/*.jpg")

        def load_and_preprocess(paths):
            tf = transforms.Compose([
                transforms.Resize((299,299)),
                transforms.PILToTensor()
            ])
            for p in paths:
                yield tf(Image.open(p).convert("RGB"))

        device = config.DEVICE
        # FID real zebras vs fake zebras

        print(f"Real zebras: {len(all_images(f'{config.TRAIN_DIR}/zebras'))}")
        print(f"Fake zebras (horse→zebra): {len(all_images('saved_images/horse2zebra'))}")
        print(f"Real horses: {len(all_images(f'{config.TRAIN_DIR}/horses'))}")
        print(f"Fake horses (zebra→horse): {len(all_images('saved_images/zebra2horse'))}")

        fid_z = FrechetInceptionDistance(feature=2048).to(device)
        for img in load_and_preprocess(all_images(f"{config.TRAIN_DIR}/zebras")):
            fid_z.update(img.unsqueeze(0).to(device), real=True)
        for img in load_and_preprocess(all_images("saved_images/horse2zebra")):
            fid_z.update(img.unsqueeze(0).to(device), real=False)
        score_h2z = fid_z.compute().item()

        # FID real horses vs fake horses
        fid_h = FrechetInceptionDistance(feature=2048).to(device)
        for img in load_and_preprocess(all_images(f"{config.TRAIN_DIR}/horses")):
            fid_h.update(img.unsqueeze(0).to(device), real=True)
        for img in load_and_preprocess(all_images("saved_images/zebra2horse")):
            fid_h.update(img.unsqueeze(0).to(device), real=False)
        score_z2h = fid_h.compute().item()

        # Inception Score on the fake sets
        is_metric = InceptionScore(splits=10).to(device)
        for img in load_and_preprocess(all_images("saved_images/horse2zebra")):
            is_metric.update(img.unsqueeze(0).to(device))
        is_h2z_mean, is_h2z_std = is_metric.compute()
        is_h2z_mean, is_h2z_std = is_h2z_mean.item(), is_h2z_std.item()

        print("\n🏁 Final metrics:")
        print(f"  FID (horse→zebra): {score_h2z:.2f}")
        print(f"  FID (zebra→horse): {score_z2h:.2f}")
        print(f"  Inception Score (horse→zebra): {is_h2z_mean:.2f} ± {is_h2z_std:.2f}")

if __name__ == "__main__":
    main()
