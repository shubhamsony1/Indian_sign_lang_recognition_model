import os
import shutil

# Define paths
source_base = r"C:\Users\ASUS\Documents\sem 8 new\Indian_sign_lang_recognition_model\DATA_COL\imgs_without_landmarks"
destination_base = r"C:\Users\ASUS\Documents\sem 8 new\Indian_sign_lang_recognition_model\data"

# Get all subfolders in the source directory
source_folders = [folder for folder in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, folder))]
source_paths = [os.path.join(source_base, folder) for folder in source_folders]

# Labels (A-Z, 0-9, NEXT, DONE, SPACE)
labels = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)] + ["NEXT", "DONE", "SPACE"]

# Create destination folder if it doesn't exist
os.makedirs(destination_base, exist_ok=True)

# Create label folders inside destination
for label in labels:
    os.makedirs(os.path.join(destination_base, label), exist_ok=True)

# Merge and rename images
for label in labels:
    image_counter = 1  # Start numbering images

    for source_path in source_paths:
        source_label_path = os.path.join(source_path, label)

        if os.path.exists(source_label_path):
            for img_name in sorted(os.listdir(source_label_path)):  # Sort to maintain order
                old_img_path = os.path.join(source_label_path, img_name)
                new_img_name = f"{image_counter}.jpg"
                new_img_path = os.path.join(destination_base, label, new_img_name)

                shutil.copy2(old_img_path, new_img_path)  # Copy with new name
                image_counter += 1  # Increment counter

print("Images successfully merged and renamed in 'data' folder.")
