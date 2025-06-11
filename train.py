"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.linalg
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def projection_loss(x_real, x_fake, k=5, r=3, eps=1e-6):
    """
    Compute local PCA projection loss between real and fake batches.
    x_real, x_fake: tensors of shape (B, C, H, W)
    Uses each sample's k nearest neighbors (in the batch) to define a local subspace.
    """
    B, C, H, W = x_real.shape

    # Need at least 2 samples and enough for r-dimensional subspace
    if B < 2 or B <= r:
        return torch.tensor(0.0, device=x_real.device)

    # Effective number of neighbors
    k_eff = min(k, B - 1)

    # Flatten images to (B, d)
    x_real_flat = x_real.view(B, -1)
    x_fake_flat = x_fake.view(B, -1)

    # Compute pairwise squared distances in the real batch
    with torch.no_grad():
        dist = (x_real_flat.unsqueeze(1) - x_real_flat.unsqueeze(0)).pow(2).sum(-1)  # (B, B)
        # include self in topk, then drop
        _, nn_idx = dist.topk(k_eff + 1, largest=False, dim=-1)  # (B, k_eff+1)
        nn_idx = nn_idx[:, 1:]  # drop self → (B, k_eff)

    losses = []
    for i in range(B):
        idx = nn_idx[i]  # neighbor indices for sample i

        # Real neighbors
        Xi = x_real_flat[idx]                     # (k_eff, d)
        Xi_cent = Xi - x_real_flat[i : i + 1]     # (k_eff, d)
        U, S, Vh = torch.linalg.svd(Xi_cent.t(), full_matrices=False)  # on (d, k_eff)
        U_r = U[:, :r]                            # (d, r)
        P_real = U_r @ U_r.t()                    # (d, d)

        # Fake neighbors (same indices)
        Xif = x_fake_flat[idx]
        Xif_cent = Xif - x_fake_flat[i : i + 1]
        Uf, Sf, Vhf = torch.linalg.svd(Xif_cent.t(), full_matrices=False)
        Uf_r = Uf[:, :r]
        P_fake = Uf_r @ Uf_r.t()

        # Frobenius norm squared
        losses.append((P_real - P_fake).pow(2).sum())

    return torch.stack(losses).mean()


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # ----------------------
        # Train Discriminators
        # ----------------------
        with torch.amp.autocast(device_type="cuda"):
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ----------------------
        # Train Generators
        # ----------------------
        with torch.amp.autocast(device_type="cuda"):
            # Adversarial losses
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # Cycle-consistency losses
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # Identity losses
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # Base CycleGAN loss
            G_loss_base = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

            # Projection loss + adaptive scaling
            eps = 1e-6
            proj = projection_loss(zebra, fake_zebra, k=5, r=3, eps=eps)
            α = G_loss_base.detach() / (proj.detach() + eps)
            α = torch.clamp(α, min=0.1, max=10.0)

            # Final generator loss
            G_loss = G_loss_base + α * proj

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Save sample outputs
        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE)

    dataset = HorseZebraDataset(
        root_horse=f"{config.TRAIN_DIR}/horses",
        root_zebra=f"{config.TRAIN_DIR}/zebras",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=f"{config.VAL_DIR}/horses",
        root_zebra=f"{config.VAL_DIR}/zebras",
        transform=config.transforms,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()
