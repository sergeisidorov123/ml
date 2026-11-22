import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "homework8"
DATA = ROOT / "dataset"
ROBOT_DATA = ROOT / "robot_detection"


def read_manifest(name):
    return json.loads((PREVIOUS / f"{name}_manifest.json").read_text(encoding="utf-8"))


def prepare_classification():
    records = []
    for class_name, manifest in [("human", "people"), ("humanoid", "robots")]:
        rows = sorted((row for row in read_manifest(manifest) if row["split"] == "base"), key=lambda row: row["id"])
        assert len(rows) == 100
        random.Random(42).shuffle(rows)
        for split, selected in [("train", rows[:70]), ("validation", rows[70:85]), ("test", rows[85:])]:
            folder = DATA / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for row in selected:
                source = PREVIOUS / row["crop"]
                target = folder / source.name
                shutil.copyfile(source, target)
                records.append({"file": target.relative_to(ROOT).as_posix(), "split": split,
                                "class": class_name, "source": row["source"], "sha256": row["sha256"]})
    assert len({row["sha256"] for row in records}) == 200
    (ROOT / "classification_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Классификация: train 140, validation 30, test 30 (поровну по классам).")


def prepare_robot_images():
    # Целые сцены B-Human. Отложенный матч целиком остаётся в тесте.
    robots = read_manifest("robots")
    base = sorted((row for row in robots if row["split"] == "base"), key=lambda row: row["id"])
    test = sorted((row for row in robots if row["split"] == "detector_test"), key=lambda row: row["id"])
    random.Random(42).shuffle(base)
    records = []
    for split, rows in [("train", base[:85]), ("val", base[85:]), ("test", test)]:
        folder = ROBOT_DATA / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = PREVIOUS / row["scene"]
            target = folder / f'{row["id"]}{source.suffix}'
            shutil.copyfile(source, target)
            records.append({"file": target.relative_to(ROOT).as_posix(), "split": split,
                            "source": row["source"], "source_frame": row["source_frame"],
                            "source_box": row["source_box"], "sha256": row["sha256"]})
    assert len(records) == len({row["sha256"] for row in records}) == 150
    (ROOT / "robot_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("YOLOv8: подготовлены 85/15/50 сцен, тест из отдельного матча.")


if __name__ == "__main__":
    prepare_classification()
    prepare_robot_images()
