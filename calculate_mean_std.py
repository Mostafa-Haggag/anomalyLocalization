import numpy as np
from PIL import Image
import os
def calculate_normalization_parameters(dataset_paths):
    """
    Calculate normalization parameters (mean and standard deviation) of multiple datasets.

    Args:
    - dataset_paths: List of paths to directories containing the datasets

    Returns:
    - mean: Mean value for each channel
    - std: Standard deviation value for each channel
    """
    # Initialize variables to accumulate mean and std
    mean = np.zeros(3)
    std = np.zeros(3)
    total_samples = 0

    # Iterate over each dataset directory
    for dataset_path in dataset_paths:
        # Iterate over each image in the dataset directory
        for filename in os.listdir(dataset_path):
            # Load image
            if not filename.lower().endswith(('.jpg', '.bmp', '.png')):
                continue

            img = np.array(Image.open(os.path.join(dataset_path, filename)))
            img = img / 255.0  # Normalize pixel values to [0, 1]

            # Calculate mean and std for each channel
            mean += np.mean(img, axis=(0, 1))
            std += np.std(img, axis=(0, 1))
            total_samples += 1

    # Calculate overall mean and std across all datasets
    mean /= total_samples
    std /= total_samples

    return mean, std
if __name__ == "__main__":
    mean, std = (
        calculate_normalization_parameters([r"D:\github_directories\foriegn\SEQ00003_MIXED_COUNTED_313"]))
    print(mean)
    print(std)