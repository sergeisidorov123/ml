"""HOG, классические классификаторы и собственный sliding-window детектор."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from skimage.feature import hog
from skimage.transform import resize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
HOG_OPTIONS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", transform_sqrt=True)


def read_photo(path):
    with Image.open(ROOT / path) as image:
        return image.convert("RGB")


def feature(image, visualize=False):
    gray = np.asarray(image.convert("L").resize((64, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return hog(gray, visualize=visualize, **HOG_OPTIONS)


def models():
    return {
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=10000, random_state=42)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, random_state=42)),
    }


def augment(image):
    return [ImageOps.mirror(image), image.rotate(7, resample=Image.Resampling.BILINEAR),
            image.rotate(-7, resample=Image.Resampling.BILINEAR), ImageEnhance.Brightness(image).enhance(0.75),
            image.filter(ImageFilter.GaussianBlur(radius=0.7))]


def iou(box, other):
    x1, y1 = max(box[0], other[0]), max(box[1], other[1])
    x2, y2 = min(box[2], other[2]), min(box[3], other[3])
    overlap = max(0, x2-x1)*max(0, y2-y1)
    area_a = max(0, box[2]-box[0])*max(0, box[3]-box[1])
    area_b = max(0, other[2]-other[0])*max(0, other[3]-other[1])
    return overlap / (area_a+area_b-overlap) if area_a+area_b-overlap else 0.0


def suppress(boxes, scores, threshold=0.3):
    order = np.argsort(scores)[::-1].tolist()
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [index for index in order if iou(boxes[best], boxes[index]) <= threshold]
    return [boxes[i] for i in keep], [float(scores[i]) for i in keep]


def fit_detector(people, robots):
    """Люди из 100 сцен + фон без размеченных людей + 100 гуманоидов."""
    X, y = [], []
    rng = np.random.default_rng(42)
    for row in people:
        image = read_photo(row["scene"])
        for box in row["boxes"]:
            sample = image.crop(box)
            X.extend([feature(sample), feature(ImageOps.mirror(sample))])
            y.extend([1, 1])
        # Отрицательные окна добавляются только из обучающих сцен.
        accepted = 0
        for _ in range(100):
            height = int(rng.integers(128, max(129, image.height)))
            width = height // 2
            if width > image.width or height > image.height:
                continue
            x = int(rng.integers(0, image.width-width+1))
            z = int(rng.integers(0, image.height-height+1))
            box = [x,z,x+width,z+height]
            if all(iou(box, b) < 0.05 for b in row["boxes"]):
                X.append(feature(image.crop(box)))
                y.append(0)
                accepted += 1
                if accepted == 5:
                    break
    for row in robots:
        image = read_photo(row["crop"])
        X.extend([feature(image), feature(ImageOps.mirror(image))])
        y.extend([0,0])
    model = LinearSVC(C=0.05, class_weight="balanced", max_iter=10000, random_state=42)
    model.fit(np.asarray(X), y)
    return model, dict(zip(*np.unique(y, return_counts=True)))


def detect(image, model, threshold=0.0):
    """Пирамида масштабов, окно 64×128, шаг 8 пикселей, NMS 0.3."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    original_h, original_w = gray.shape
    scale = max(min(1.0, 512/max(gray.shape)), 128/original_h, 64/original_w)
    boxes, scores = [], []
    while int(original_h*scale) >= 128 and int(original_w*scale) >= 64:
        height, width = int(original_h*scale), int(original_w*scale)
        scaled = resize(gray, (height, width), anti_aliasing=True, preserve_range=True).astype(np.float32)
        blocks = hog(scaled, feature_vector=False, **HOG_OPTIONS)
        coordinates = [(y,x) for y in range(blocks.shape[0]-14) for x in range(blocks.shape[1]-6)]
        if coordinates:
            for start in range(0,len(coordinates),512):
                current = coordinates[start:start+512]
                features = np.asarray([blocks[y:y+15,x:x+7].reshape(-1) for y,x in current])
                output = model.decision_function(features)
                for (y,x), score in zip(current,output):
                    if score > threshold:
                        sx,sy=original_w/width,original_h/height
                        boxes.append([x*8*sx,y*8*sy,(x*8+64)*sx,(y*8+128)*sy])
                        scores.append(float(score))
        scale /= 1.25
    return suppress(boxes, scores)


def match_boxes(truth, predicted):
    """Каждая рамка участвует не более чем в одном совпадении при IoU≥0.5."""
    pairs = sorted([(iou(t,p),i,j) for i,t in enumerate(truth) for j,p in enumerate(predicted)], reverse=True)
    taken_truth,taken_pred=set(),set()
    for overlap,i,j in pairs:
        if overlap>=0.5 and i not in taken_truth and j not in taken_pred:
            taken_truth.add(i)
            taken_pred.add(j)
    best_iou=[max([iou(t,p) for p in predicted],default=0) for t in truth]
    return len(taken_truth),len(predicted)-len(taken_pred),len(truth)-len(taken_truth),best_iou
