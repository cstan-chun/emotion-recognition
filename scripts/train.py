import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import load_config
from data_utils import EmotionDataset, collate_emotion_batch
from emotion.model import EmotionModel


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(model, loader, optimizer, criterion, device, mode: str):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for mfcc, frames, labels in tqdm(loader, desc="Train"):
        mfcc = mfcc.to(device)
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if mode == "multimodal":
            logits = model.forward_multimodal(mfcc, frames)
        elif mode == "audio":
            logits = model.forward_audio_only(mfcc)
        else:
            logits = model.forward_visual_only(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device, mode: str):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mfcc, frames, labels in tqdm(loader, desc="Val"):
            mfcc = mfcc.to(device)
            frames = frames.to(device)
            labels = labels.to(device)
            if mode == "multimodal":
                logits = model.forward_multimodal(mfcc, frames)
            elif mode == "audio":
                logits = model.forward_audio_only(mfcc)
            else:
                logits = model.forward_visual_only(frames)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--stage", choices=["visual", "audio", "joint", "finetune"],
                        default="visual")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = EmotionModel(cfg).to(device)

    train_ds = EmotionDataset(args.data_dir, split="train")
    val_ds = EmotionDataset(args.data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                              shuffle=True, collate_fn=collate_emotion_batch,
                              num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                            shuffle=False, collate_fn=collate_emotion_batch,
                            num_workers=cfg.data.num_workers)

    t_cfg = cfg.training
    if args.stage == "visual":
        mode = "visual"
        for name, param in model.audio_encoder.named_parameters():
            param.requires_grad = False
        for name, param in model.fusion.named_parameters():
            param.requires_grad = False
        for name, param in model.classifier.named_parameters():
            param.requires_grad = False
    elif args.stage == "audio":
        mode = "audio"
        for name, param in model.visual_encoder.named_parameters():
            param.requires_grad = False
        for name, param in model.fusion.named_parameters():
            param.requires_grad = False
        for name, param in model.classifier.named_parameters():
            param.requires_grad = False
    elif args.stage == "joint":
        mode = "multimodal"
        for name, param in model.visual_encoder.named_parameters():
            param.requires_grad = False
        for name, param in model.audio_encoder.named_parameters():
            param.requires_grad = False
    elif args.stage == "finetune":
        mode = "multimodal"

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=t_cfg.weight_decay,
    )
    criterion = FocalLoss(alpha=t_cfg.focal_alpha, gamma=t_cfg.focal_gamma)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device, mode)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion,
                                             device, mode)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt_dir = t_cfg.checkpoint_dir
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            }, os.path.join(ckpt_dir, f"best_{args.stage}.pt"))
            print(f"Saved checkpoint with F1: {val_f1:.4f}")


if __name__ == "__main__":
    main()
