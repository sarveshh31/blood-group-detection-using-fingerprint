import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(BLOOD_GROUPS)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BloodGroupDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None):
        self.samples = []
        self.transform = transform
        dataset_path = Path(dataset_dir)

        for cls_name in BLOOD_GROUPS:
            cls_dir = dataset_path / cls_name
            if not cls_dir.exists():
                print(f"[WARNING] Class folder not found: {cls_dir}")
                continue
            label = CLASS_TO_IDX[cls_name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(img_path), label))

        print(f"[Dataset] Loaded {len(self.samples)} samples across {len(BLOOD_GROUPS)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(dataset_dir: str, batch_size: int = 64, use_train_aug: bool = True, num_workers: int = 4):
    transform = get_train_transforms() if use_train_aug else get_eval_transforms()
    dataset = BloodGroupDataset(dataset_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, [label for _, label in dataset.samples]