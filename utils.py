import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    # (optionally) use weights_only=True if you're on a PyTorch version that supports it:
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)

    old_state = checkpoint["state_dict"]
    new_state = {}

    for k, v in old_state.items():
        # If the checkpoint was saved from a model where `self.initial` was a single Conv2d,
        # its keys will look like "initial.weight"/"initial.bias". Our current Generator
        # defines `self.initial = nn.Sequential(nn.Conv2d(...), ...)`, so the first Conv2d layer
        # is now named "initial.0". We need to insert that ".0" here.
        if k.startswith("initial.") and not k.startswith("initial.0."):
            # e.g. "initial.weight" → "initial.0.weight"
            new_key = k.replace("initial.", "initial.0.", 1)
            new_state[new_key] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Make sure optimizer uses the desired lr instead of whatever was in the checkpoint:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False