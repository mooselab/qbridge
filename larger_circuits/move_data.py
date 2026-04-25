import os
import shutil

src_root = "./data/data/evaluation_data"
dst_root = "./data/evaluation_data"

for folder_name in os.listdir(src_root):
    src_folder = os.path.join(src_root, folder_name)

    if not os.path.isdir(src_folder):
        continue

    dst_folder = os.path.join(dst_root, folder_name)
    os.makedirs(dst_folder, exist_ok=True)

    for item_name in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item_name)
        dst_item = os.path.join(dst_folder, item_name)

        if os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)
            print(f"Copied file: {src_item} -> {dst_item}")
        elif os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            print(f"Copied folder: {src_item} -> {dst_item}")

print("Done.")