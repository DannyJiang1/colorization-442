import os
import requests
from zipfile import ZipFile
import shutil


import os
import shutil


def centralize_images(source_dir, target_dir):
    """
    Centralizes all images from nested subdirectories with 'images' folders into a single target directory.

    Args:
        source_dir (str): The root directory containing the nested image folders.
        target_dir (str): The directory where all images will be centralized.

    Returns:
        None
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Traverse the source directory
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        # Check if 'images' folder exists in the current class directory
        images_dir = os.path.join(class_path, "images")
        print(images_dir)
        if os.path.isdir(images_dir):
            for file in os.listdir(images_dir):
                # Only process image files
                if file.endswith((".JPEG")):
                    src_path = os.path.join(images_dir, file)
                    dest_path = os.path.join(target_dir, file)

                    # Handle potential duplicate filenames
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(
                                target_dir, f"{base}_{counter}{ext}"
                            )
                            counter += 1

                    # Move or copy the image
                    shutil.move(
                        src_path, dest_path
                    )  # Use shutil.copy() to copy instead of move
                    print(f"Moved: {src_path} -> {dest_path}")


def download_tiny_imagenet(destination_folder="data"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_filename = os.path.join(destination_folder, "tiny-imagenet-200.zip")

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Download the dataset
    print("Downloading Tiny ImageNet dataset...")
    response = requests.get(url, stream=True)
    with open(zip_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")

    # Extract the dataset
    print("Extracting Tiny ImageNet dataset...")
    with ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(destination_folder)
    print("Extraction complete.")

    # Remove the zip file
    os.remove(zip_filename)
    print("Cleaned up zip file.")


# Call the function
download_tiny_imagenet()

# Paths
source_directory = "data/tiny-imagenet-200/train"  # Source directory
target_directory = (
    "data/tiny-imagenet-200/centralized_train_images"  # Target directory
)

# Call the function
centralize_images(source_directory, target_directory)
