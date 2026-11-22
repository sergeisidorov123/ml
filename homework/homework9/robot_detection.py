import argparse
import json
from pathlib import Path
import zipfile

from PIL import Image
import torch
import yaml
from ultralytics import YOLO, settings

from box_metrics import intersection_over_union, summarize


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "robot_detection"
RESULTS = ROOT / "results"
PRETRAINED = ROOT / "pretrained" / "yolov8n.pt"
CONFIDENCE = 0.25
ROBOT_NAMES = {"robot", "robot_fallen", "humaoid", "humaoid_fallen"}


def manifest():
    return json.loads((ROOT / "robot_manifest.json").read_text(encoding="utf-8"))


def normalized_box(label):
    left, top = max(0, label["x"]), max(0, label["y"])
    right = min(100, label["x"] + label["width"])
    bottom = min(100, label["y"] + label["height"])
    if left >= right or top >= bottom:
        return None
    return [(left + right) / 200, (top + bottom) / 200,
            (right - left) / 100, (bottom - top) / 100]


def prepare_source_labels(source_folder):
    records = manifest()
    annotations = {}
    for source in sorted({row["source"] for row in records}):
        path = Path(source_folder) / source.rsplit("/", 1)[-1]
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".json"))
            labels = json.loads(archive.read(name))
        annotations[source] = {row["image"]: row["label"] for row in labels}
    counts = {}
    for row in records:
        path = ROOT / row["file"]
        folder = DATA / "labels" / row["split"]
        folder.mkdir(parents=True, exist_ok=True)
        boxes = [normalized_box(label) for label in annotations[row["source"]][row["source_frame"]]
                 if label["class"] in ROBOT_NAMES]
        lines = ["0 " + " ".join(f"{value:.8f}" for value in box) for box in boxes if box is not None]
        if not lines:
            raise ValueError(f"Нет рамок роботов на {path.name}")
        (folder / (path.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts[row["split"]] = counts.get(row["split"], 0) + len(lines)
    config = {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
              "names": {0: "humanoid_robot"}}
    (DATA / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Исходные рамки B-Human:", counts, flush=True)


def truth_boxes(path, width, height):
    labels = DATA / "labels" / "test" / (path.stem + ".txt")
    truth = []
    for line in labels.read_text(encoding="utf-8").splitlines():
        _, x, y, w, h = map(float, line.split())
        truth.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return truth


def evaluate(model, name, custom):
    model_names = model.names
    if not custom:
        assert "humanoid_robot" not in model_names.values()
    results = []
    for row in manifest():
        if row["split"] != "test":
            continue
        path = ROOT / row["file"]
        with Image.open(path) as image:
            width, height = image.size
        result = model.predict(str(path), device="cpu", imgsz=416, conf=CONFIDENCE, verbose=False)[0]
        boxes = result.boxes.xyxy.tolist()
        labels = [model_names[int(index)] for index in result.boxes.cls.tolist()]
        results.append({"scene": row["file"], "truth": truth_boxes(path, width, height),
                        "boxes": boxes, "labels": labels, "scores": result.boxes.conf.tolist()})
    geometry = summarize(results)
    semantic_rows = [
        {**row, "boxes": [box for box, label in zip(row["boxes"], row["labels"]) if label == "humanoid_robot"]}
        for row in results
    ]
    semantics = summarize(semantic_rows)
    mistakes = {}
    for row in results:
        for box, label in zip(row["boxes"], row["labels"]):
            if label != "humanoid_robot" and any(intersection_over_union(box, t) >= 0.5 for t in row["truth"]):
                mistakes[label] = mistakes.get(label, 0) + 1
    file = RESULTS / f"{name}.json"
    file.write_text(json.dumps({"confidence": CONFIDENCE, "device": "cpu", "class_aware": semantics,
                               "any_class": geometry, "wrong_classes_overlapping_robots": mistakes,
                               "predictions": results}, indent=2), encoding="utf-8")
    print(name, semantics, "Any class", geometry, "Wrong classes", mistakes, flush=True)


def classification_sample_predictions(model):
    output = []
    files = sorted((ROOT / "dataset" / "test").glob("*/*.jpg"))
    assert len(files) == 30
    for path in files:
        result = model.predict(str(path), device="cpu", imgsz=416, conf=CONFIDENCE, verbose=False)[0]
        output.append({"scene": path.relative_to(ROOT).as_posix(), "class": path.parent.name,
                       "boxes": result.boxes.xyxy.tolist(),
                       "labels": [model.names[int(i)] for i in result.boxes.cls.tolist()]})
    (RESULTS / "yolo_prepared_classification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Предобученная YOLOv8n проверена на 30 подготовленных кадрах ДЗ-8.", flush=True)


def train(data_yaml, epochs, name):
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    # Физический путь добавляется только на время запуска.
    config["path"] = str(data_yaml.parent.resolve())
    config_file = RESULTS / f"{name}_data.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model = YOLO(str(PRETRAINED))
    model.train(data=str(config_file), device="cpu", epochs=epochs, imgsz=320, batch=8,
                workers=0, seed=42, freeze=10, plots=False, amp=False,
                project=str(RESULTS), name=name, exist_ok=True)
    best = RESULTS / name / "weights" / "best.pt"
    evaluate(YOLO(str(best)), name, custom=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["labels", "prepared", "baseline", "train"])
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(min(3, torch.get_num_threads()))
    settings.update({"sync": False, "tensorboard": False, "wandb": False, "mlflow": False})
    if args.action == "labels":
        if args.sources is None:
            parser.error("--sources обязателен для labels")
        prepare_source_labels(args.sources)
    elif args.action == "prepared":
        classification_sample_predictions(YOLO(str(PRETRAINED)))
    elif args.action == "baseline":
        evaluate(YOLO(str(PRETRAINED)), "yolo_pretrained", custom=False)
    else:
        data = args.data or DATA / "data.yaml"
        name = "yolo_roboflow" if args.data else "yolo_source_labels"
        train(data.resolve(), args.epochs, name)


if __name__ == "__main__":
    main()
