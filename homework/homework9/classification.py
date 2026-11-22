from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")


def simple_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 4 * 4, 2),
    )


def vgg11():
    # Сверточная часть VGG-11 (конфигурация A), выходная голова для двух классов.
    layers = []
    channels = 3
    for width in [64, "pool", 128, "pool", 256, 256, "pool", 512, 512, "pool", 512, 512, "pool"]:
        if width == "pool":
            layers.append(nn.MaxPool2d(2))
        else:
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)])
            channels = width
    layers.extend([
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2),
    ])
    model = nn.Sequential(*layers)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
    return model


def loaders():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    data = {name: datasets.ImageFolder(ROOT / "dataset" / name, transform=transform)
            for name in ["train", "validation", "test"]}
    assert all(dataset.classes == ["human", "humanoid"] for dataset in data.values())
    return data, {
        name: DataLoader(dataset, batch_size=8, shuffle=name == "train", num_workers=0,
                         generator=torch.Generator().manual_seed(42))
        for name, dataset in data.items()
    }


def run_epoch(model, batches, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0
    labels, predicted = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for images, target in batches:
            logits = model(images.to(DEVICE))
            loss = loss_fn(logits, target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(target)
            labels.extend(target.tolist())
            predicted.extend(logits.detach().argmax(1).tolist())
    return {"loss": loss_sum / len(labels),
            "accuracy": sum(a == b for a, b in zip(labels, predicted)) / len(labels),
            "target": labels, "prediction": predicted}


def train_classifiers(epochs=8):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    for name, build, lr in [("cnn", simple_cnn, 0.001), ("vgg11", vgg11, 0.0003)]:
        torch.manual_seed(42)
        data, batches = loaders()
        model = build().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = []
        best = (float("-inf"), float("-inf"))
        for epoch in range(1, epochs + 1):
            train = run_epoch(model, batches["train"], optimizer)
            val = run_epoch(model, batches["validation"])
            history.append({"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                            "val_loss": val["loss"], "val_accuracy": val["accuracy"]})
            key = (val["accuracy"], -val["loss"])
            if key > best:
                best = key
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
            print(f"{name} epoch {epoch}/{epochs} | loss={train['loss']:.3f} | val_acc={val['accuracy']:.3f}", flush=True)
        model.load_state_dict(best_state)
        test = run_epoch(model, batches["test"])
        test["files"] = [Path(file).relative_to(ROOT).as_posix() for file, _ in data["test"].samples]
        torch.save(best_state, output / f"{name}.pt")
        payload = {"architecture": name, "history": history, "best_epoch": best_epoch, "test": test,
                   "device": DEVICE.type, "classes": data["test"].classes}
        (output / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name} test accuracy={test['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    train_classifiers()
