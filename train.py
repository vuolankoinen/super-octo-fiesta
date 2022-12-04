import sys
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch

sys.path.insert(1, "./")
import config

device = torch.device("cuda")


class Trainer:
    def __init__(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def accuracy(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        top_p, top_class = y_pred.topk(1, dim=1)
        equals = top_class == y_true.view(*top_class.shape)
        return torch.mean(equals.type(torch.FloatTensor))

    def train_batch_loop(self, model, train_loader):

        train_loss = 0.0
        train_acc = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += self.accuracy(logits, labels)

        return train_loss / len(train_loader), train_acc / len(train_loader)

    def val_batch_loop(self, model, val_loader):

        val_loss = 0.0
        val_acc = 0.0

        for images, true_labels in tqdm(val_loader):

            images = images.to(device)
            true_labels = true_labels.to(device)

            logits = model(images)
            loss = self.criterion(logits, true_labels)

            val_loss += loss.item()
            val_acc += self.accuracy(logits, true_labels)

        return val_loss / len(val_loader), val_acc / len(val_loader)

    def fit(self, model, train_loader, val_loader, epochs, name="", start_epoch=0):
        timestamp = "-".join(
            [
                (lambda x: str(x) if x > 9 else f"0{x}")(th)
                for th in datetime.datetime.now().timetuple()[:6]
            ]
        )

        val_min_loss = np.Inf
        val_max_acc = 0

        writer = SummaryWriter(f"{config.LOG_DIR}/{timestamp}-log-{name}")

        iters_since_best = 0

        for i in range(start_epoch, epochs):

            model.train()
            avg_train_loss, avg_train_acc = self.train_batch_loop(model, train_loader)

            model.eval()
            avg_val_loss, avg_val_acc = self.val_batch_loop(model, val_loader)

            if avg_val_loss <= val_min_loss:
                print(
                    f"Validation loss decreased from {val_min_loss} to {avg_val_loss}"
                )
                torch.save(
                    model,
                    f"{config.MODEL_DIR}/{timestamp}-{name}-val-{avg_val_acc*100:.0f}-epoch-{i+1}.pytorch",
                )
                val_min_loss = avg_val_loss
                iters_since_best = 0
            if avg_val_acc > val_max_acc:
                print(
                    f"Validation accuracy increased from {val_max_acc} to {avg_val_acc}"
                )
                torch.save(
                    model,
                    f"{config.MODEL_DIR}/{timestamp}-{name}-val-{avg_val_acc*100:.0f}-epoch-{i+1}-acc-based.pytorch",
                )
                val_max_acc = avg_val_acc
                iters_since_best = 0
            else:
                iters_since_best += 1

            print(
                f"Epoch: {i + 1} Train Loss: {avg_train_loss:.6f} Train Accuracy: {avg_train_acc:.6f}"
            )
            print(
                f"Epoch: {i + 1} Valid Loss: {avg_val_loss:.6f} Valid Acc: {avg_val_acc:.6f}"
            )

            writer.add_scalars(
                "Accuracy", {"training": avg_train_acc, "validation": avg_val_acc}, i
            )
            writer.add_scalars(
                "Loss", {"training": avg_train_loss, "validation": avg_val_loss}, i
            )

            if (
                i > 1
                and avg_val_acc < 0.55
                and avg_train_acc < 0.55
                and not val_min_loss == avg_val_loss
            ):
                print("It seems there's no learning happening. :( ABORTING")
                break

            if i > 15 and iters_since_best > 14:
                print(
                    f"Stopping early, since no improvement in {iters_since_best} iterations"
                )
                break

        writer.close()
        torch.save(
            model,
            f"{config.MODEL_DIR}/{timestamp}-{name}-val-{avg_val_acc*100:.0f}-epoch-{i+1}-last-one.pytorch",
        )
