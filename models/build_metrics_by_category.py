
"""Расчёт метрик моделей по категориям фотографий."""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "golden_set/golden_set.jsonl"
META = ROOT / "data/metadata_all_400.csv"
PRED_DIR = ROOT / "models/evaluation_predictions"
OUTPUT = ROOT / "models/metrics_by_category.csv"

MODELS = {
    "dino": ("dino_conf_0.35.jsonl", "0.35"),
    "owl": ("owl_conf_0.50.jsonl", "0.5"),
    "qwen": ("qwen.jsonl", ""),
}


def read_jsonl(path):
    records = {}

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            photo_id = str(item["photo_id"])

            if photo_id in records:
                raise ValueError(
                    f"Duplicate photo_id {photo_id}: {path}"
                )

            records[photo_id] = item

    return records


def iou(a, b):
    x1 = max(a["left"], b["left"])
    y1 = max(a["top"], b["top"])
    x2 = min(
        a["left"] + a["width"],
        b["left"] + b["width"],
    )
    y2 = min(
        a["top"] + a["height"],
        b["top"] + b["height"],
    )

    intersection = (
        max(0.0, x2 - x1) * max(0.0, y2 - y1)
    )

    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def matches(predictions, targets, threshold):
    # Максимальное паросочетание рамок.
    # Одна эталонная рамка сопоставляется
    # не более чем с одним предсказанием.

    edges = [
        [
            j
            for j, target in enumerate(targets)
            if iou(pred, target) >= threshold
        ]
        for pred in predictions
    ]

    assigned = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue

            visited.add(j)

            if j not in assigned or augment(
                assigned[j], visited
            ):
                assigned[j] = i
                return True

        return False

    return sum(
        augment(i, set())
        for i in range(len(predictions))
    )


def empty_stats():
    return {
        "photos": 0,
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "tp_iou_075": 0,
        "false_boxes": 0,
        "errors": 0,
    }


def main():
    gold = read_jsonl(GOLD)

    with META.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        category_by_id = {
            str(row["photo_id"]): row["category"]
            for row in csv.DictReader(file)
        }

    if set(gold) != set(category_by_id):
        raise ValueError(
            "Photo IDs in metadata and Golden Set do not match"
        )

    rows = []

    for model, (filename, confidence) in MODELS.items():
        predictions = read_jsonl(PRED_DIR / filename)

        if set(gold) != set(predictions):
            raise ValueError(
                f"{model}: predictions do not cover "
                "all Golden Set photos"
            )

        counts = defaultdict(empty_stats)

        for photo_id, truth_record in gold.items():
            record = predictions[photo_id]

            truth = truth_record["boxes"]
            error = bool(record.get("error"))
            pred = record.get("boxes") or []

            tp = matches(pred, truth, 0.5)
            tp_075 = matches(pred, truth, 0.75)

            category = category_by_id[photo_id]
            stats = counts[category]

            stats["photos"] += 1
            stats["TP"] += tp
            stats["TN"] += int(
                not truth and not pred and not error
            )
            stats["FP"] += len(pred) - tp
            stats["FN"] += len(truth) - tp
            stats["tp_iou_075"] += tp_075

            stats["false_boxes"] += (
                len(pred) if not truth else 0
            )
            stats["errors"] += int(error)

        for category in sorted(counts):
            s = counts[category]

            tp = s["TP"]
            fp = s["FP"]
            fn = s["FN"]

            precision = (
                tp / (tp + fp) if tp + fp else 0
            )
            recall = (
                tp / (tp + fn) if tp + fn else 0
            )

            denominator = 2 * tp + fp + fn

            f1_05 = (
                2 * tp / denominator
                if denominator else 0
            )

            f1_075 = (
                2 * s["tp_iou_075"] / denominator
                if denominator else 0
            )

            rows.append({
                "model": model,
                "confidence": confidence,
                "category": category,
                **s,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_iou_05": round(f1_05, 6),
                "f1_iou_075": round(f1_075, 6),
            })

    columns = [
        "model",
        "confidence",
        "category",
        "photos",
        "TP",
        "TN",
        "FP",
        "FN",
        "tp_iou_075",
        "false_boxes",
        "errors",
        "precision",
        "recall",
        "f1_iou_05",
        "f1_iou_075",
    ]

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUTPUT}; rows: {len(rows)}")
    print(
        "Categories:",
        len(set(row["category"] for row in rows)),
    )


if __name__ == "__main__":
    main()
