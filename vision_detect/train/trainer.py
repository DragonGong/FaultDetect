from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import torch
from typing import Callable, Optional
import os


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_fn: Callable,
            optimizer: optim.Optimizer,
            pred_fn: Callable,
            device: torch.device,
            save_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.pred_fn = pred_fn
        self.device = device
        self.save_dir = save_dir
        self.best_val_loss = float('inf')
        self.best_model_state = None
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        with tqdm(self.train_loader, desc="Train", leave=False) as t:
            for images, labels in t:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)
                t.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Val", leave=False) as t:
                for images, labels in t:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    running_loss += loss.item() * images.size(0)
                    correct += self.pred_fn(outputs, labels)
                    t.set_postfix(loss=loss.item())

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = 100. * correct / len(self.val_loader.dataset)
        return val_loss, val_acc

    def save_checkpoint(self, state_dict: dict, path: str):
        torch.save(state_dict, path)
        print(f"Saved model to {path}")

    def train(self, num_epochs: int = 50):
        print(f"Starting training for {num_epochs} epochs...\n")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_path = os.path.join(self.save_dir, "best.pth")
                self.save_checkpoint(self.best_model_state, best_path)
                print(f"Saved best model at epoch {epoch + 1} (Val Loss: {val_loss:.4f})")
            print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%\n")
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model weights.")
        final_path = os.path.join(self.save_dir, "final.pth")
        self.save_checkpoint(self.model.state_dict(), final_path)
        return self.model
