import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

def get_image_paths(folder):
    exts = (".png", ".jpg", ".jpeg")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def load_images(image_paths, transform):
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        yield transform(img)

def compute_metrics(real_dir, fake_dir, device="cuda"):
    real_paths = get_image_paths(real_dir)
    fake_paths = get_image_paths(fake_dir)

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise RuntimeError("No images found in one of the folders.")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    print(f"[INFO] Found {len(real_paths)} real and {len(fake_paths)} fake images.")

    # ------------------ FID ------------------
    print("[INFO] Computing Frechet Inception Distance (FID)...")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for img in tqdm(load_images(real_paths, transform), total=len(real_paths), desc="Real"):
        fid.update(img.unsqueeze(0).to(device), real=True)

    for img in tqdm(load_images(fake_paths, transform), total=len(fake_paths), desc="Fake"):
        fid.update(img.unsqueeze(0).to(device), real=False)

    fid_score = fid.compute().item()

    # ------------------ IS ------------------
    print("[INFO] Computing Inception Score (IS)...")
    is_metric = InceptionScore(splits=10).to(device)

    for img in tqdm(load_images(fake_paths, transform), total=len(fake_paths), desc="IS"):
        is_metric.update(img.unsqueeze(0).to(device))

    is_mean, is_std = is_metric.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()

    return fid_score, is_mean, is_std

def main():
    parser = argparse.ArgumentParser(description="Evaluate FID and IS for generated images.")
    parser.add_argument("--real", required=True, help="Path to real images folder (e.g. real zebras)")
    parser.add_argument("--fake", required=True, help="Path to generated images folder (e.g. fake zebras)")
    args = parser.parse_args()

    fid, is_mean, is_std = compute_metrics(args.real, args.fake)

    print("\n📊 Evaluation Results:")
    print(f"FID: {fid:.2f} ↓")
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f} ↑")

if __name__ == "__main__":
    main()
