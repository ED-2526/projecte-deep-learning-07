"""
Entrenament del CNN sobre QuickDraw amb integració de Weights & Biases.

Ús bàsic (un sol experiment):
    python train.py --epochs 10 --lr 0.001 --batch_size 128

Ús amb sweep (W&B llança variants automàticament):
    wandb sweep sweep.yaml
    wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
"""

import os
import json
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageDraw

import wandb


# ---------------------------------------------------------------------------
# 1. DATASET
# ---------------------------------------------------------------------------
class QuickDrawDataset(Dataset):
    """Carrega els .ndjson del Quickdraw simplified i renderitza en bitmap."""

    def __init__(self, dataset_dir, categories_file, max_items_per_class=2500,
                 image_size=64, line_width=5, transform=None):
        self.image_size = image_size
        self.line_width = line_width
        self.transform = transform
        self.data = []
        self.labels = []

        with open(categories_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(f"Carregant {len(self.classes)} categories "
              f"(max {max_items_per_class} per classe)...")
        for cls in self.classes:
            filepath = os.path.join(dataset_dir, f"full_simplified_{cls}.ndjson")
            if not os.path.exists(filepath):
                print(f"  [AVÍS] No trobada {filepath}")
                continue
            count = 0
            with open(filepath, "r") as f:
                for line in f:
                    if count >= max_items_per_class:
                        break
                    record = json.loads(line)
                    if record.get("recognized", False):
                        self.data.append(record["drawing"])
                        self.labels.append(self.class_to_idx[cls])
                        count += 1
        print(f"Dataset carregat: {len(self.data)} imatges, "
              f"{len(self.classes)} classes.")

    def _draw_strokes_to_image(self, strokes):
        image = Image.new("L", (256, 256), color=0)
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            x, y = stroke[0], stroke[1]
            for i in range(len(x) - 1):
                draw.line((x[i], y[i], x[i + 1], y[i + 1]),
                          fill=255, width=self.line_width)
        return image.resize((self.image_size, self.image_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self._draw_strokes_to_image(self.data[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------
class QuickDrawCNN(nn.Module):
    """CNN parametrizable: profunditat, dropout i FC-size configurables."""

    def __init__(self, num_classes, image_size=64, dropout=0.3, fc_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # 3 maxpool -> image_size/8
        flat_size = 128 * (image_size // 8) ** 2
        self.fc1 = nn.Linear(flat_size, fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# 3. ENTRENAMENT
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    top3_correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            # top-3 accuracy
            _, pred3 = outputs.topk(3, dim=1)
            top3_correct += (pred3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    return loss_sum / total, 100 * correct / total, 100 * top3_correct / total


def train(args):
    # --- W&B init ---
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=args.run_name,
        tags=args.tags,
    )
    cfg = wandb.config  # permet sweeps reescriure els paràmetres

    set_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Executant a: {device}")
    wandb.log({"device": str(device)})

    # --- Dades ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full = QuickDrawDataset(
        dataset_dir=cfg.dataset_dir,
        categories_file=cfg.categories_file,
        max_items_per_class=cfg.max_items_per_class,
        image_size=cfg.image_size,
        line_width=cfg.line_width,
        transform=transform,
    )
    train_size = int(0.9 * len(full))
    val_size = len(full) - train_size
    train_set, val_set = random_split(
        full, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # --- Model ---
    model = QuickDrawCNN(
        num_classes=len(full.classes),
        image_size=cfg.image_size,
        dropout=cfg.dropout,
        fc_size=cfg.fc_size,
    ).to(device)

    wandb.watch(model, log="gradients", log_freq=200)

    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(cfg.optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step,
                                          gamma=cfg.lr_gamma)

    # --- Train loop ---
    best_acc = 0.0
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        t0 = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, pred = torch.max(outputs, 1)
            running_correct += (pred == labels).sum().item()
            running_total += labels.size(0)
            global_step += 1

            if (i + 1) % cfg.log_every == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + (i + 1) / len(train_loader),
                }, step=global_step)

        train_loss = running_loss / running_total
        train_acc = 100 * running_correct / running_total
        val_loss, val_acc, val_top3 = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - t0

        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/top3_acc": val_top3,
            "epoch_time_s": epoch_time,
            "epoch": epoch + 1,
        }, step=global_step)

        print(f"[{epoch+1}/{cfg.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
              f"top3={val_top3:.2f}% ({epoch_time:.1f}s)")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = os.path.join(cfg.out_dir, f"best_{run.id}.pth")
            os.makedirs(cfg.out_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
            wandb.run.summary["best_val_acc"] = best_acc
            if cfg.log_artifact:
                art = wandb.Artifact(f"model-{run.id}", type="model")
                art.add_file(ckpt)
                wandb.log_artifact(art)

    print(f"Millor val_acc: {best_acc:.2f}%")
    wandb.finish()


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Dades
    p.add_argument("--dataset_dir", default="simplified_dataset")
    p.add_argument("--categories_file", default="categories.txt")
    p.add_argument("--max_items_per_class", type=int, default=2500)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--line_width", type=int, default=5)
    # Model
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--fc_size", type=int, default=512)
    # Optim
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_step", type=int, default=2)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--log_artifact", action="store_true",
                   help="Puja el checkpoint com a artefacte W&B")
    # W&B
    p.add_argument("--wandb_project", default="quickdraw-xn-g07")
    p.add_argument("--wandb_entity", default=None,
                   help="Equip de W&B (deixar buit per usar el teu user)")
    p.add_argument("--run_name", default=None)
    p.add_argument("--tags", nargs="*", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
