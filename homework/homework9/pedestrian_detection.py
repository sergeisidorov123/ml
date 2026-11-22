import argparse
import json
import random
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import pil_to_tensor

from box_metrics import summarize


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "homework8"
RESULTS = ROOT / "results"
DEVICE = torch.device("cpu")
THRESHOLD = 0.35


def split_scenes():
    people = json.loads((SOURCE / "people_manifest.json").read_text(encoding="utf-8"))
    train = [row for row in people if row["split"] == "base"]
    held_out = sorted((row for row in people if row["split"] == "detector_test"), key=lambda row: row["id"])
    random.Random(42).shuffle(held_out)
    assert len(train) == 100 and len(held_out) == 50
    assert len({row["sha256"] for row in train + held_out}) == 150
    return {"train": train, "validation": held_out[:25], "test": held_out[25:]}


class PennFudan(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = SOURCE / record["scene"]
        with Image.open(path) as image:
            pixels = pil_to_tensor(image.convert("RGB")).float() / 255
        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.long)}
        return pixels, target, record["scene"]


def collate(batch):
    return tuple(zip(*batch))


def build_model(name):
    if name == "ssdlite":
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            rpn_pre_nms_top_n_train=300, rpn_post_nms_top_n_train=150,
            rpn_pre_nms_top_n_test=150, rpn_post_nms_top_n_test=100,
        )
    model.backbone.requires_grad_(False)
    return model.to(DEVICE)


@torch.inference_mode()
def predict(model, batches):
    model.eval()
    results = []
    for photos, targets, paths in batches:
        outputs = model(list(photos))
        for output, target, path in zip(outputs, targets, paths):
            keep = (output["labels"] == 1) & (output["scores"] >= THRESHOLD)
            results.append({"scene": path, "truth": target["boxes"].tolist(),
                            "boxes": output["boxes"][keep].tolist(), "scores": output["scores"][keep].tolist()})
    return results


def run(name, epochs):
    torch.set_num_threads(min(3, torch.get_num_threads()))
    torch.manual_seed(42)
    RESULTS.mkdir(exist_ok=True)
    torch.hub.set_dir(str(ROOT / "pretrained"))
    splits = split_scenes()
    batches = {
        split: DataLoader(PennFudan(data), batch_size=2, shuffle=split == "train",
                          collate_fn=collate, num_workers=0,
                          generator=torch.Generator().manual_seed(42))
        for split, data in splits.items()
    }
    model = build_model(name)
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad),
                                lr=0.001, momentum=0.9, weight_decay=0.0005)
    history = []
    best_key = (float("-inf"), float("-inf"))
    best_path = RESULTS / f"{name}.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        model.backbone.eval()
        total = 0.0
        for step, (photos, targets, _) in enumerate(batches["train"], 1):
            loss = sum(model(list(photos), list(targets)).values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {epoch}/{step}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            total += loss.item()
            if step % 20 == 0:
                print(f"{name} epoch {epoch} step {step}/{len(batches['train'])}", flush=True)
        val = summarize(predict(model, batches["validation"]))
        history.append({"epoch": epoch, "train_loss": total / len(batches["train"]), "validation": val})
        key = (val["f1_at_iou50"], val["mean_best_iou"])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(f"{name} epoch {epoch}/{epochs} | loss {history[-1]['train_loss']:.3f} | val F1 {key[0]:.3f}", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    test = predict(model, batches["test"])
    stats = summarize(test)
    with (RESULTS / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump({"device": "cpu", "confidence": THRESHOLD, "best_epoch": best_epoch, "history": history,
                   "test": stats, "predictions": test,
                   "split": {key: [r["id"] for r in rows] for key, rows in splits.items()}}, handle, indent=2)
    print(name, stats, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["ssdlite", "faster_rcnn"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run(args.model, args.epochs)
