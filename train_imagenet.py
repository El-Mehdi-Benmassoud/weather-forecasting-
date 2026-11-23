"""Train a ResNet-18 from scratch on an ImageNet-style folder dataset."""

from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Dataset directory containing train/ and val/ folders")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/imagenet"))
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA automatic mixed precision (often faster on GPUs)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is not available.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset: datasets.ImageFolder, batch_size: int, workers: int,
                shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
              device: torch.device, optimizer: torch.optim.Optimizer | None,
              scaler: torch.amp.GradScaler | None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.inference_mode()
        with grad_context:
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if scaler is not None else nullcontext()
            )
            with autocast_context:
                logits = model(images)
                loss = loss_fn(logits, labels)

            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        batch_size = labels.size(0)
        loss_sum += loss.detach().item() * batch_size
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return loss_sum / sample_count, correct / sample_count


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("AMP is only enabled on CUDA; continuing without it.")

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_dir = args.data / "train"
    val_dir = args.data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected ImageNet folders at {train_dir} and {val_dir}; see README.md."
        )
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("train/ and val/ must contain the same class subfolders.")
    if not train_data:
        raise ValueError("The training folder contains no images.")
    if not val_data:
        raise ValueError("The validation folder contains no images.")

    train_loader = make_loader(train_data, args.batch_size, args.workers, True, device)
    val_loader = make_loader(val_data, args.batch_size, args.workers, False, device)

    # weights=None means every model parameter starts with a random initialization.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    args.output.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0
    print(f"Device: {device}; classes: {len(train_data.classes)}")
    print(f"Training images: {len(train_data):,}; validation images: {len(val_data):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, loss_fn, device, None, None
        )
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": train_data.classes,
            "best_val_accuracy": max(best_accuracy, val_accuracy),
        }
        torch.save(checkpoint, args.output / "last.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint["best_val_accuracy"] = best_accuracy
            torch.save(checkpoint, args.output / "best.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.2%} | "
            f"best {best_accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
