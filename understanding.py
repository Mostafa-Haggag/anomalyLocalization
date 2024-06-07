import os
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt


class CenterCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        width, height = image.size
        new_width, new_height = self.crop_size

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        return image.crop((left, top, right, bottom))

class MVTecAD(data.Dataset):
    """Dataset class for the MVTecAD dataset."""

    def __init__(self, image_dir, transform=None, crop_size=(400, 1936)):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.crop_size = crop_size
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.bmp', '.png'))]

    def __getitem__(self, index):
        """Return one image"""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB

        original_image = image.copy()
        cropped_image = CenterCrop(self.crop_size)(image)

        if self.transform is not None:
            original_image = self.transform(original_image)
            cropped_image = self.transform(cropped_image)
        return original_image, cropped_image

    def __len__(self):
        """Return the number of images."""
        return len(self.image_paths)
# Function to plot images
# def plot_images(images, n_images=4):
#     fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
#     for i, img in enumerate(images[:n_images]):
#         img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
#         axes[i].imshow(img, cmap='gray')
#         axes[i].axis('off')
#     plt.show()
# Example usage:
# Function to plot original and cropped images side by side
def plot_images(original_images, cropped_images):
    n_images = len(original_images)
    fig, axes = plt.subplots(n_images, 2, figsize=(20, 10 * n_images))

    for i in range(n_images):
        original_img = original_images[i].permute(1, 2, 0).numpy()  # Convert to HWC format
        cropped_img = cropped_images[i].permute(1, 2, 0).numpy()  # Convert to HWC format

        axes[i, 0].imshow(original_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Original Image")

        axes[i, 1].imshow(cropped_img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Cropped Image")

    plt.show()

if __name__ == "__main__":
    image_dir = r"D:\github_directories\foriegn\SEQ00004_MIXED_FOREIGN_PARTICLE"
    crop_size = (600, 1936)  # Desired crop size

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = MVTecAD(image_dir=image_dir, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through the dataset
    for original_images, transformed_images in dataloader:
        plot_images(original_images, transformed_images)

