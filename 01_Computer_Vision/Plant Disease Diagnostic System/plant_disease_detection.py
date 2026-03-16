# %%
import torch
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_mobilenet_v3_large_fpn,
    MobileNet_V3_Large_Weights,
)
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from glob import glob
import os
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# %%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, dset_dir, transform):
        super().__init__()
        self.dset_dir = dset_dir
        self.dset_imgs = glob(os.path.join(dset_dir, "images/*.*"))
        self.transform = transform

    def __getitem__(self, index):
        img_dir = self.dset_imgs[index]
        img = Image.open(img_dir).convert("RGB")
        img_base = os.path.basename(img_dir)
        img_name, ext = os.path.splitext(img_base)
        img_label_dir = os.path.join(self.dset_dir, "labels", f"{img_name}.txt")
        boxes = []
        clses = []
        with open(img_label_dir, encoding="utf-8") as f:
            for line in f.readlines():
                cls, cx, cy, w, h = map(float, line.split(" "))
                cls = int(cls) + 1
                x1 = (cx - w / 2) * 300
                y1 = (cy - h / 2) * 300
                x2 = (cx + w / 2) * 300
                y2 = (cy + h / 2) * 300
                boxes.append([x1, y1, x2, y2])
                clses.append(cls)

        target = {
            "boxes": torch.tensor(boxes).float(),
            "labels": torch.tensor(clses).long(),
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dset_imgs)


transform = T.Compose(
    [
        T.Resize((300, 300)),
        T.ToTensor(),
        T.RandomAutocontrast(),
    ]
)

tr_dset = CustomDataset(
    r"data\plant_disease_dataset",
    transform=transform,
)

test_dset = CustomDataset(
    r"data\plant_disease_dataset",
    transform=transform,
)


def collate_fn(batch):
    return list(zip(*batch))


tr_dataloader = DataLoader(tr_dset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(
    test_dset, batch_size=8, shuffle=True, collate_fn=collate_fn
)


# %%

model = (
    fasterrcnn_mobilenet_v3_large_fpn(
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=3,
    )
    .to(device)
    .train()
)

# for p in model.backbone.parameters():
#     p.requires_grad = False


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=6e-4,
)


# %%
metric = MeanAveragePrecision("xyxy")
n_epochs = 20
for e in range(n_epochs):
    model.train()
    epoch_loss = 0
    for images, target in tr_dataloader:
        images = [t.to(device) for t in images]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        optimizer.zero_grad()
        losses = model(images, target)
        losses = sum(list(losses.values()))
        epoch_loss += losses.item()
        losses.backward()
        optimizer.step()

    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, target in test_dataloader:
            images = [t.to(device) for t in images]
            pred = model(images)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]
            metric.update(pred, target)

    r = metric.compute()
    mAP = r["map"].item()
    mAP_50 = r["map_50"].item()

    print(f"Epoch {e+1}/{n_epochs}")
    print(
        f"Train Loss: {epoch_loss/len(tr_dataloader):.4f} | mAP: {mAP:.4f} | mAP@50: {mAP_50:.4f}"
    )



# %%

# torch.save(model.state_dict(), "weights.pth")
