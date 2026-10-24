
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP


# 1. Configuration
BATCH_SIZE = 128
DEVICE = torch.device("cpu")

MODEL_PATH = Path(
    "models/experiment_1/best_mlp.pth"
)

RESULT_PATH = Path(
    "results/experiment_1/test_metrics.json"
)


# 2. Prepare test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 3. Load selected model
model = MLP(
    hidden1=256,
    hidden2=128
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

criterion = nn.CrossEntropyLoss()


# 4. Evaluate test dataset
total_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size


# 5. Calculate metrics
test_loss = total_loss / total
test_accuracy = 100 * correct / total


# 6. Print results
print("\nTEST RESULTS")
print("----------------------------")
print(f"Total images: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {total - correct}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2f}%")


# 7. Save results
metrics = {
    "experiment": 1,
    "total_images": total,
    "correct_predictions": correct,
    "wrong_predictions": total - correct,
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
}

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(RESULT_PATH, "w") as file:
    json.dump(metrics, file, indent=2)

print(f"\nResults saved to: {RESULT_PATH}")
