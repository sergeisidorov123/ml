"""Validate and import an unaugmented Roboflow YOLOv8 export."""

import argparse
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent
CLASS = "humanoid_robot"


def original_key(filename):
    stem = PurePosixPath(filename).stem
    if ".rf." in stem:
        stem = re.sub(r"[_-](?:png|jpe?g)$", "", stem.split(".rf.", 1)[0], flags=re.I)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def validate_labels(raw, filename):
    lines = raw.decode("utf-8").splitlines()
    boxes = 0
    for line in lines:
        if not line.strip():
            continue
        label, x, y, width, height = map(float, line.split())
        if label != 0 or not (
            width > 0 and height > 0 and
            -1e-4 <= x-width/2 <= x+width/2 <= 1.0001 and
            -1e-4 <= y-height/2 <= y+height/2 <= 1.0001
        ):
            raise ValueError(f"Неверная разметка в {filename}")
        boxes += 1
    if boxes == 0:
        raise ValueError(f"Из экспорта пропали все рамки: {filename}")
    return boxes


def import_export(export, destination):
    records = json.loads((ROOT / "robot_manifest.json").read_text(encoding="utf-8"))
    expected = {original_key(row["file"]): row for row in records if row["split"] in {"train", "val"}}
    found = set()
    prepared = []
    with ZipFile(export) as archive:
        members = [PurePosixPath(name) for name in archive.namelist()]
        configs = [yaml.safe_load(archive.read(entry)) for entry in archive.infolist()
                   if PurePosixPath(entry.filename).name == "data.yaml"]
        if not configs or any(config.get("names") not in ([CLASS], {0: CLASS}, {"0": CLASS})
                              for config in configs):
            raise ValueError(f"Экспорт должен содержать единственный класс {CLASS}")
        for path in members:
            if path.parent.name != "images" or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            key = original_key(path.name)
            record = expected.get(key)
            if key in found or record is None:
                raise ValueError(f"Лишний или повторяющийся кадр: {path.name}")
            split = record["split"]
            known_splits = {"train": "train", "valid": "val", "val": "val"}
            if known_splits.get(path.parent.parent.name) != split:
                raise ValueError(f"Roboflow изменил назначение кадра {path.name}: ожидался {split}")
            image_bytes = archive.read(str(path))
            with Image.open(BytesIO(image_bytes)) as image, Image.open(ROOT / record["file"]) as source:
                if image.size != source.size:
                    raise ValueError(f"Экспорт изменил размер {path.name}. Отключите Resize.")
                image.verify()
            label_path = path.parent.parent / "labels" / path.with_suffix(".txt").name
            label = archive.read(str(label_path))
            boxes = validate_labels(label, path.name)
            found.add(key)
            prepared.append((split, Path(record["file"]).stem + path.suffix.lower(), image_bytes, label, boxes))
    if found != expected.keys():
        raise ValueError(f"Не найдены исходные кадры: {sorted(expected.keys() - found)}")

    destination.mkdir()
    for split, name, pixels, label, _ in prepared:
        images, labels = (destination / kind / split for kind in ("images", "labels"))
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        (images / name).write_bytes(pixels)
        (labels / Path(name).with_suffix(".txt")).write_bytes(label)
    (destination / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val", "names": {0: CLASS}},
                       sort_keys=False), encoding="utf-8",
    )
    print(f"Проверены {len(prepared)} кадров и {sum(row[-1] for row in prepared)} рамок.")
    print(destination / "data.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="ZIP, скачанный из Roboflow в формате YOLOv8")
    parser.add_argument("--output", type=Path, default=ROOT / "roboflow_dataset")
    args = parser.parse_args()
    import_export(args.export, args.output)
