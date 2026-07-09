import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.metrics import f1_score, recall_score

from src.models.CAMC import CAMC
from src.utils.env import HF_TOKEN
from src.utils import paths
from src.utils.plot import plot_loss_acc

# 0=entailment, 1=neutral, 2=contradiction
CONTRADICTION_LABEL = 2


def config_dict(cfg):
    values = {}
    for key, value in vars(cfg).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def prepare_data_dir(cfg):
    # train.slurm copies the dataset to flash ($TMP_SHARED) before python
    data_dir = paths.resolve_path(cfg.data_dir)

    tmp_shared = os.getenv("TMP_SHARED")
    if tmp_shared:
        staged_dir = Path(tmp_shared) / cfg.staged_data_name
        if staged_dir.exists():
            print(f"Using staged dataset: {staged_dir}")
            return staged_dir
        print(f"No staged dataset at {staged_dir}; using {data_dir}")

    return data_dir


def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(data_dir, cfg, tokenizer, device):
    ds = load_from_disk(data_dir)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def preprocess(examples):
        inputs = tokenizer(
            examples["hypothesis"],
            truncation=True,
            max_length=128
        )
        inputs["pixel_values"] = [
            image_transform(img.convert("RGB"))
            for img in examples["image"]
        ]
        inputs["labels"] = examples["label"]
        return inputs

    def get_cleaned_dataset(set_name):
        split = ds[set_name]
        if cfg.subset > 0:
            split = split.select(range(min(cfg.subset, len(split))))
        split = split.filter(lambda label: label != -1, input_columns=["label"])
        split.set_transform(preprocess)
        return split

    train_ds = get_cleaned_dataset("train")
    val_ds = get_cleaned_dataset("validation")

    text_collator = DataCollatorWithPadding(tokenizer)

    def multimodal_collate_fn(features):
        images = [f.pop("pixel_values") for f in features]
        batch = text_collator(features)          # pad input_ids / attention_mask
        batch["pixel_values"] = torch.stack(images)
        return batch

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_ds, val_ds


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            img_feature = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(img_feature, input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_loss = total_loss / len(loader)
    val_acc = float((all_preds == all_labels).mean())
    val_macro_f1 = f1_score(all_labels, all_preds, average="macro")
    val_contradiction_recall = recall_score(
        all_labels, all_preds,
        labels=[CONTRADICTION_LABEL], average=None, zero_division=0
    )[0]
    return val_loss, val_acc, val_macro_f1, val_contradiction_recall


def train(model, optimizer, criterion, train_loader, val_loader,
          cfg, out_dir, device):
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    csv_path = out_dir / "metrics.csv"

    start_epoch = 0
    best_val_f1 = 0.0
    best_epoch = 0

    # Resume: if last.pt exists, continue from it (re-sbatch after timeout)
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        print(f"Resumed from {last_path}: start epoch {start_epoch+1}, "
              f"best val macro-F1 {best_val_f1:.4f}")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "train_acc", "val_loss",
                "val_acc", "val_macro_f1", "val_contradiction_recall"])

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        num_train_samples = 0

        for batch in train_loader:
            img_feature = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(img_feature, input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # Clip gradient spikes so one bad batch cannot blow up the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            _, predicted = torch.max(logits, 1)
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            num_train_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / num_train_samples

        val_loss, val_acc, val_f1, contra_recall = evaluate(
            model, val_loader, criterion, device)

        # best.pt: saved whenever val macro-F1 hits a new high
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)

        # Write to disk at the end of every epoch, so nothing is lost if killed
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        }, last_path)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}",
                f"{val_f1:.6f}", f"{contra_recall:.6f}"])

        print(f"Epoch {epoch+1}/{cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Macro-F1: {val_f1:.4f} | "
              f"Contra Recall: {contra_recall:.4f} | "
              f"Best Val F1: {best_val_f1:.4f}")

    return best_val_f1, best_epoch


def main(config, config_name):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(torch.__version__)
    print("Using device:", device)
    print("Experiment:", config.exp_name)

    data_dir = prepare_data_dir(config)
    out_dir = paths.output_dir(config.results_dir, config.exp_name)

    os.makedirs(out_dir, exist_ok=True)
    # Write config.json once at startup: the full experiment config
    with open(out_dir / "config.json", "w") as f:
        json.dump({**config_dict(config), "config_name": config_name,
                   "output_dir": str(out_dir),
                   "data_dir": str(data_dir), "device": str(device)},
                  f, indent=2)

    # Seed before creating DataLoaders so shuffle order is reproducible
    set_seed(config.seed, device)

    tokenizer_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", **tokenizer_kwargs)
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        data_dir, config, tokenizer, device)
    print(f"train/val size: {len(train_ds)}/{len(val_ds)}")

    model = CAMC(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_f1, best_epoch = train(
        model, optimizer, criterion, train_loader, val_loader,
        config, out_dir, device)
    print(f"Best val macro-F1 {best_val_f1:.4f} @ epoch {best_epoch}")

    # Plot from the full metrics.csv history, so curves stay complete after resume
    with open(out_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    if rows:
        metrics = {k: [float(r[k]) for r in rows]
                   for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
        plot_loss_acc(metrics, len(rows), out_dir)


if __name__ == "__main__":
    raise SystemExit(
        "Run an experiment entrypoint, e.g. "
        "python -m src.configs.camc_seed42"
    )
