import os
import shutil
import random


def merge_magnifications(raw_path, working_path):

    root_folder = os.path.join(
        raw_path,
        "dataset_cancer_v1",
        "classificacao_binaria"
    )

    benign_dest = os.path.join(working_path, "benign_all")
    malignant_dest = os.path.join(working_path, "malignant_all")

    os.makedirs(benign_dest, exist_ok=True)
    os.makedirs(malignant_dest, exist_ok=True)

    for mag_folder in os.listdir(root_folder):
        mag_path = os.path.join(root_folder, mag_folder)

        if os.path.isdir(mag_path):
            for label in ["benign", "malignant"]:
                label_path = os.path.join(mag_path, label)

                if os.path.isdir(label_path):
                    for file in os.listdir(label_path):
                        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                            src = os.path.join(label_path, file)
                            dest_folder = benign_dest if label == "benign" else malignant_dest
                            dest = os.path.join(dest_folder, file)

                            base, ext = os.path.splitext(file)
                            count = 1
                            while os.path.exists(dest):
                                dest = os.path.join(dest_folder, f"{base}_{count}{ext}")
                                count += 1

                            shutil.copy2(src, dest)

    return benign_dest, malignant_dest


def split_dataset(benign_folder, malignant_folder, output_root, train_ratio=0.8):

    paths = {
        "train/benign": os.path.join(output_root, "train", "benign"),
        "train/malignant": os.path.join(output_root, "train", "malignant"),
        "test/benign": os.path.join(output_root, "test", "benign"),
        "test/malignant": os.path.join(output_root, "test", "malignant"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    def shuffle_and_split(src_folder, train_dest, test_dest):

        images = [f for f in os.listdir(src_folder)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]

        random.seed(42)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy2(os.path.join(src_folder, img),
                         os.path.join(train_dest, img))

        for img in test_imgs:
            shutil.copy2(os.path.join(src_folder, img),
                         os.path.join(test_dest, img))

    shuffle_and_split(benign_folder,
                      paths["train/benign"],
                      paths["test/benign"])

    shuffle_and_split(malignant_folder,
                      paths["train/malignant"],
                      paths["test/malignant"])

    return os.path.join(output_root, "train"), os.path.join(output_root, "test")