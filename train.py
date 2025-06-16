"""
Training for CycleGAN + Projection Loss
"""

import torch
import torch.linalg
import os
import matplotlib.pyplot as plt
from dataset import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from discriminator_model import Discriminator
from generator_model import Generator

# For tracking losses
generator_losses = []
projection_losses = []

def projection_loss(x_real, x_fake, k=5, r=3, eps=1e-6):
    B, C, H, W = x_real.shape
    if B < 2 or B <= r:
        print(f"[⚠️ Warning] Batch size {B} too small for r={r} projection.")
        return torch.tensor(0.0, device=x_real.device)

    k_eff = min(k, B - 1)
    if k_eff < k:
        print(f"[⚠️ Warning] Adjusting k from {k} to {k_eff} due to small batch size.")

    x_real_flat = x_real.view(B, -1)
    x_fake_flat = x_fake.view(B, -1)

    with torch.no_grad():
        dist = (x_real_flat.unsqueeze(1) - x_real_flat.unsqueeze(0)).pow(2).sum(-1)
        _, nn_idx = dist.topk(k_eff + 1, largest=False, dim=-1)
        nn_idx = nn_idx[:, 1:]

    losses = []
    for i in range(B):
        idx = nn_idx[i]
        Xi = x_real_flat[idx]
        mu = Xi.mean(dim=0, keepdim=True)
        Xi_cent = Xi - mu
        U, _, _ = torch.linalg.svd(Xi_cent.t(), full_matrices=False)
        U_r = U[:, :r]
        P_real = U_r @ U_r.t()

        Xif = x_fake_flat[idx]
        muf = Xif.mean(dim=0, keepdim=True)
        Xif_cent = Xif - muf
        Uf, _, _ = torch.linalg.svd(Xif_cent.t(), full_matrices=False)
        Uf_r = Uf[:, :r]
        P_fake = Uf_r @ Uf_r.t()

        losses.append((P_real - P_fake).pow(2).sum())

    return torch.stack(losses).mean()


def save_triplet(input_img, generated_img, cycle_img, filename):
    imgs = torch.cat([input_img, generated_img, cycle_img], dim=0)
    grid = make_grid(imgs, nrow=3, normalize=True, value_range=(-1, 1))
    save_image(grid, filename)

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
                proj_z = projection_loss(zebra, fake_zebra)
                proj_h = projection_loss(horse, fake_horse)
                proj = proj_z + proj_h
                α = G_base.detach() / (proj.detach() + 1e-6)
                α = torch.clamp(α, min=0.1, max=10.0)
                G_loss = G_base + α * proj * 2.0  # Amplify projection contribution

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


# rest of the code remains unchanged...


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
    fixed_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    fixed_zebra, fixed_horse = next(iter(fixed_loader))
    fixed_zebra, fixed_horse = fixed_zebra.to(config.DEVICE), fixed_horse.to(config.DEVICE)
    # training loader
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
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

if __name__ == "__main__":
    main()
