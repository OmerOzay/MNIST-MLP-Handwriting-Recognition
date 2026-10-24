

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, hidden1=256, hidden2=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(784, hidden1),
            nn.ReLU(),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, 10)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)


if __name__ == "__main__":

    model = MLP()

    print("Model architecture:")
    print(model)

    dummy_image = torch.randn(1, 1, 28, 28)

    output = model(dummy_image)

    print("Output shape:", output.shape)

    predicted_digit = torch.argmax(output, dim=1).item()

    print("Predicted digit:", predicted_digit)
