import os
from PIL import Image


def resize_images_in_folder(folder_path, max_width):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Create a folder for resized images
    resized_folder_path = os.path.join(folder_path, "resized")
    os.makedirs(resized_folder_path, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if os.path.isfile(file_path) and filename.lower().endswith(("png", "jpg", "jpeg", "gif", "bmp")):
            with Image.open(file_path) as img:
                # Calculate the new height to maintain the aspect ratio
                width_percent = max_width / float(img.size[0])
                new_height = int((float(img.size[1]) * float(width_percent)))

                # Resize the image
                img = img.resize((max_width, new_height), resample=Image.Resampling.LANCZOS)

                # Save the resized image to the new folder
                resized_image_path = os.path.join(resized_folder_path, filename)
                img.save(resized_image_path)
                print(f"Resized image saved as {resized_image_path}")


# Example usage
folder_path = "../../yolo-testing/datasets/exDark-yolo/images/val"
max_width = 800
resize_images_in_folder(folder_path, max_width)
