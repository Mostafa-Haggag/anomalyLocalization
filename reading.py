import os
from PIL import Image, ImageChops

def create_black_mask(image_path, save_path, mask_size):
    """
    Create a black mask with the same size as the reference mask.
    """
    black_mask = Image.new('L', mask_size, 0)  # 'L' mode for grayscale, 0 for black
    black_mask.save(save_path)

def main():
    image_list_file = r'C:\Users\Jacopo\Downloads\foreing_particles\ImageSets\Segmentation\default.txt'
    mask_folder = r'C:\Users\Jacopo\Downloads\foreing_particles\creating'
    mask_extension = '.png'  # Extension for mask files

    # Read the list of image names
    with open(image_list_file, 'r') as file:
        image_names = file.read().splitlines()

    # Find a reference mask to get the size
    reference_mask_path = None
    for filename in os.listdir(mask_folder):
        if filename.endswith(mask_extension):
            reference_mask_path = os.path.join(mask_folder, filename)
            break

    if not reference_mask_path:
        raise ValueError("No reference mask found in the folder")

    # Get the size of the reference mask
    reference_mask = Image.open(reference_mask_path)
    mask_size = reference_mask.size

    # Check each image and create black masks for missing ones
    for image_name in image_names:
        mask_path = os.path.join(mask_folder, os.path.splitext(image_name)[0] + mask_extension)
        if not os.path.exists(mask_path):
            print(f"Creating black mask for {image_name}")
            create_black_mask(image_name, mask_path, mask_size)

if __name__ == "__main__":
    main()