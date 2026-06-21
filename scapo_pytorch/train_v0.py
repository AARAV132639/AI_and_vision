import torch
from torch.utils.data import DataLoader

from datasets.synthetic_laptop import SyntheticLaptopDataset

from models.scapo_v0 import SCAPOv0

from losses.cycle_loss import CycleLoss
from losses.segmentation_loss import SegmentationLoss
from losses.kp_seg_loss import KeypointSegLoss


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4


dataset = SyntheticLaptopDataset(
    num_samples=5000
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = SCAPOv0(
    num_parts=4
).to(DEVICE)

cycle_loss_fn = CycleLoss()

seg_loss_fn = SegmentationLoss()

kp_loss_fn = KeypointSegLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in loader:

        points = batch["points"].float().to(
            DEVICE
        )

        canonical = batch[
            "canonical"
        ].float().to(
            DEVICE
        )

        optimizer.zero_grad()

        out = model(points)

        deformed = out["deformed"]

        weights = out["weights"]

        keypoints = out["keypoints"]

        #
        # losses
        #

        loss_cycle = cycle_loss_fn(
            deformed,
            canonical
        )

        loss_seg = seg_loss_fn(
            weights,
            points,
            keypoints
        )

        loss_kp = kp_loss_fn(
            points,
            weights,
            keypoints
        )

        loss = (
            10.0 * loss_cycle
            + 1.0 * loss_seg
            + 1.0 * loss_kp
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss
        / len(loader)
    )

    print(
        f"Epoch {epoch+1:03d}"
        f" | Loss {avg_loss:.4f}"
    )