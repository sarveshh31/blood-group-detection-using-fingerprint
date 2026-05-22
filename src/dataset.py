import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(BLOOD_GROUPS)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
NUM_CLASSES   = len(BLOOD_GROUPS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BloodGroupDataset(Dataset):
    """
    Reads images from:
        dataset_root/
            A+/  A-/  B+/  B-/  AB+/  AB-/  O+/  O-/
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples  : list of (image_path_str, label_int)
            transform: torchvision transform pipeline
        """
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_samples(dataset_dir: str) -> list:
    """Scan dataset_dir and return list of (path, label_idx) for all valid images."""
    samples = []
    dataset_path = Path(dataset_dir)

    for cls_name in BLOOD_GROUPS:
        cls_dir = dataset_path / cls_name
        if not cls_dir.exists():
            print(f"[WARNING] Class folder missing: {cls_dir}")
            continue
        label = CLASS_TO_IDX[cls_name]
        found = 0
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in BloodGroupDataset.VALID_EXTS:
                samples.append((str(img_path), label))
                found += 1
        print(f"  {cls_name:<6}  →  {found:>4} images")

    print(f"\n  Total samples collected: {len(samples)}\n")
    return samples