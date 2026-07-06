from transformers import AutoTokenizer, DataCollatorWithPadding

from datasets import load_from_disk
import pandas as pd
import os
import random
import json
import time
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

print(torch.__version__)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, reading HF_TOKEN from environment only")

hf_token = os.getenv("HF_TOKEN")

ds = load_from_disk("./snli-ve")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

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
    split = ds[set_name].select(range(1000))
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

# macOS + Jupyter 下 num_workers>0 容易因 multiprocessing 报错，本地调试用 0
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    collate_fn=multimodal_collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    collate_fn=multimodal_collate_fn,
    num_workers=0
)


ROOT = './'
CPPath = ROOT + 'checkpoints/'
ResultPath = ROOT + 'results/'
os.makedirs(CPPath, exist_ok=True)
os.makedirs(ResultPath, exist_ok=True)

def train(model, num_epochs):
    model = model.to(device)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0

    # 每个 epoch 实时追加写 log，训练中断也有记录；文件名带时间戳，多次运行不覆盖
    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_path = ResultPath + f'train_{run_name}.log'
    lr = optimizer.param_groups[0]['lr']
    with open(log_path, 'a') as f:
        f.write(f"run: {run_name} | device: {device} | epochs: {num_epochs} | "
                f"lr: {lr} | seed: {seed} | "
                f"train/val size: {len(train_ds)}/{len(val_ds)}\n")

    for epoch in range(num_epochs):
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
            optimizer.step()

            _, predicted = torch.max(logits, 1)

            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            num_train_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / num_train_samples

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        num_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                img_feature = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(img_feature, input_ids, attention_mask)
                loss = criterion(logits, labels)

                _, predicted = torch.max(logits, 1)
                val_loss += loss.item()
                val_correct += (predicted == labels).sum().item()
                num_val_samples += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / num_val_samples

        # save checkpoint while best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), CPPath + 'best.pt')

        # record for plot
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        log_line = (f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Best Val Acc: {best_val_acc:.4f}")
        print(log_line)
        with open(log_path, 'a') as f:
            f.write(log_line + '\n')

    metrics = {
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "val_loss": val_loss_list,
        "val_acc": val_acc_list,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }
    with open(ResultPath + f'metrics_{run_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

model = CAMC(hidden_dim=768).to(device) 
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
num_epochs = 2
metrics = train(model, num_epochs)

plot_loss_acc(metrics, len(metrics["train_loss"]))