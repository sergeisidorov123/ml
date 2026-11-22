"""Pack the fixed train/validation split for use in Roboflow Annotate."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "robot_detection"
MANIFEST = ROOT / "robot_manifest.json"


def pack(destination):
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = {"train": ("train", 85, 196), "val": ("valid", 15, 36)}
    # COCO puts names, image sizes and pixel-coordinate boxes in one file per split.
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for source_split, (export_split, image_count, box_count) in splits.items():
            images, annotations = [], []
            for record in records:
                if record["split"] != source_split:
                    continue
                path = ROOT / record["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError(f"Содержимое кадра изменилось: {path.name}")
                with Image.open(path) as image:
                    width, height = image.size
                image_id = len(images) + 1
                images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})
                label = DATA / "labels" / source_split / f"{path.stem}.txt"
                for line in label.read_text(encoding="utf-8").splitlines():
                    class_id, x, y, w, h = map(float, line.split())
                    if class_id != 0 or w <= 0 or h <= 0:
                        raise ValueError(f"Неверная рамка: {label.name}")
                    x1, y1 = max(0, x - w / 2) * width, max(0, y - h / 2) * height
                    x2, y2 = min(1, x + w / 2) * width, min(1, y + h / 2) * height
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        raise ValueError(f"Пустая рамка: {label.name}")
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                        "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
                archive.write(path, f"{export_split}/{path.name}")
            if len(images) != image_count or len(annotations) != box_count:
                raise ValueError(f"Число кадров или рамок не соответствует манифесту: {source_split}")
            coco = {"info": {"description": "B-Human robot scenes for Homework 9"},
                    "images": images, "annotations": annotations,
                    "categories": [{"id": 1, "name": "humanoid_robot", "supercategory": "robot"}]}
            archive.writestr(f"{export_split}/_annotations.coco.json", json.dumps(coco, indent=2))


if __name__ == "__main__":
    target = ROOT / "roboflow_upload.zip"
    pack(target)
    print(target)
