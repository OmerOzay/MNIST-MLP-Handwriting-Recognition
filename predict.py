
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter

from model import MLP


IMAGE_PATH = Path("handwriting/student_number.jpg")
MODEL_PATH = Path("models/experiment_1/best_mlp.pth")
OUTPUT_DIR = Path("results/handwriting")

THRESHOLD = 100


def prepare_digits(image_path):

    # Load and convert to grayscale
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.filter(ImageFilter.MedianFilter(3))

    pixels = np.asarray(image)

    # Detect dark handwriting on white paper
    ink = pixels < THRESHOLD

    # Find groups of columns containing ink
    min_pixels = max(2, int(image.height * 0.005))
    active_columns = np.flatnonzero(
        ink.sum(axis=0) >= min_pixels
    )

    if len(active_columns) == 0:
        raise ValueError("No handwritten digits detected.")

    # Separate digits at empty columns
    gaps = np.where(np.diff(active_columns) > 1)[0]

    starts = np.r_[
        active_columns[0],
        active_columns[gaps + 1]
    ]

    ends = np.r_[
        active_columns[gaps] + 1,
        active_columns[-1] + 1
    ]

    digit_images = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for start, end in zip(starts, ends):

        region = ink[:, start:end]

        if end - start < 3 or region.sum() < 20:
            continue

        rows = np.flatnonzero(region.any(axis=1))

        if len(rows) == 0:
            continue

        # Crop to the digit boundaries
        cropped = region[rows[0]:rows[-1] + 1, :]

        digit = Image.fromarray(
            (cropped * 255).astype(np.uint8)
        )

        # Resize while preserving aspect ratio
        scale = min(
            20 / digit.width,
            20 / digit.height
        )

        new_width = max(1, round(digit.width * scale))
        new_height = max(1, round(digit.height * scale))

        digit = digit.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Center the digit using its intensity center
        values = np.asarray(digit, dtype=np.float32)
        yy, xx = np.indices(values.shape)

        mass = values.sum()

        center_x = (xx * values).sum() / mass
        center_y = (yy * values).sum() / mass

        left = int(np.clip(
            round(13.5 - center_x),
            0,
            28 - new_width
        ))

        top = int(np.clip(
            round(13.5 - center_y),
            0,
            28 - new_height
        ))

        # Create MNIST-like 28x28 image
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(digit, (left, top))

        digit_images.append(canvas)

        filename = OUTPUT_DIR / (
            f"digit_{len(digit_images):02d}.png"
        )

        canvas.save(filename)

    if not digit_images:
        raise ValueError("No valid digits found.")

    # Save a larger preview of the detected digits
    preview = Image.new(
        "L",
        (28 * len(digit_images), 28),
        0
    )

    for index, digit in enumerate(digit_images):
        preview.paste(digit, (index * 28, 0))

    preview = preview.resize(
        (preview.width * 8, preview.height * 8),
        Image.Resampling.NEAREST
    )

    preview.save(OUTPUT_DIR / "digits_preview.png")

    return digit_images


def predict_digits(digit_images):

    # Create the selected MLP architecture
    model = MLP(hidden1=256, hidden2=128)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    tensors = []

    for digit in digit_images:

        array = np.asarray(
            digit,
            dtype=np.float32
        ) / 255.0

        tensor = torch.from_numpy(array).unsqueeze(0)

        # Apply the same normalization used in training
        tensor = (tensor - 0.1307) / 0.3081

        tensors.append(tensor)

    batch = torch.stack(tensors)

    with torch.inference_mode():
        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predictions = probabilities.max(dim=1)

    return predictions.tolist(), confidence.tolist()


def main():

    digits = prepare_digits(IMAGE_PATH)

    print("Detected digits:", len(digits))

    predictions, confidences = predict_digits(digits)

    predicted_number = "".join(
        str(digit) for digit in predictions
    )

    print("\nINDIVIDUAL PREDICTIONS")

    for index, (digit, confidence) in enumerate(
        zip(predictions, confidences),
        start=1
    ):
        print(
            f"Digit {index}: {digit} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

    print("\nPredicted number:", predicted_number)

    actual_number = input(
        "\nEnter actual student number "
        "(or press Enter to skip): "
    ).strip()

    if actual_number:

        if not actual_number.isascii() or not actual_number.isdigit():
            print("Please enter digits only.")

        elif len(actual_number) != len(predictions):
            print("Digit count mismatch.")
            print("Check the digits_preview.png file.")

        else:
            correct = sum(
                a == b
                for a, b in zip(predicted_number, actual_number)
            )

            accuracy = 100 * correct / len(actual_number)

            print(f"Correct digits: {correct}")
            print(f"Handwriting accuracy: {accuracy:.2f}%")

    print("\nPreview saved to:")
    print(OUTPUT_DIR / "digits_preview.png")


if __name__ == "__main__":
    main()
