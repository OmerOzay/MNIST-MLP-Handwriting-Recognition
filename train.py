
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MLP


# 1. Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

HIDDEN1 = 128
HIDDEN2 = 64

torch.manual_seed(42)
device = torch.device("cpu")

print("Device:", device)


# 2. Create output directories
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# 3. Prepare MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 4. Model, loss function ve optimizer 
model = MLP(HIDDEN1, HIDDEN2).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 5. Training gecmisini saklama
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

best_val_loss = float("inf")
best_epoch = 0

start_time = time.perf_counter()


# 6. Training dongusu
for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Record training metrics
        batch_size = labels.size(0)

        train_loss_sum += loss.item() * batch_size
        train_total += batch_size

        predictions = outputs.argmax(dim=1)
        train_correct += (
            predictions == labels
        ).sum().item()

    train_loss = train_loss_sum / train_total
    train_acc = 100 * train_correct / train_total

    # 7. Validation
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            val_loss_sum += loss.item() * batch_size
            val_total += batch_size

            predictions = outputs.argmax(dim=1)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = val_loss_sum / val_total
    val_acc = 100 * val_correct / val_total

    # 8. Save metrics
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    # Validation loss kullanılarak en iyi model kaydedilir
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            "models/best_mlp.pth"
        )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )


# 9. Training duration
training_time = time.perf_counter() - start_time

print("\nTraining completed!")
print(f"Training time: {training_time:.2f} seconds")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")


# 10. Save experiment metrics to CSV
with open("results/history.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ])

    for i in range(EPOCHS):
        writer.writerow([
            i + 1,
            history["train_loss"][i],
            history["val_loss"][i],
            history["train_acc"][i],
            history["val_acc"][i]
        ])


# 11. Create loss graph
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig("results/loss.png", dpi=150)
plt.close()


# 12. Create accuracy graph
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_acc"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("results/accuracy.png", dpi=150)
plt.close()

print("Results saved to results/")
print("Best model saved to models/best_mlp.pth")
