
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

# MNIST donusumleri
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Egitim verileri indiriliyor
full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Egitim ve dogrulama kumeleri 
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)

# Test verileri indiriliyor
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Veri seti bilgisi goruntuleme
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# Ornek inceleme
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Image label:", label)
