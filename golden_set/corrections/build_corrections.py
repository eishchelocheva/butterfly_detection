import csv
import json
from collections import Counter
from pathlib import Path

review_path = Path(
    "review/results/anonymized/review_main_numbered.jsonl"
)
golden_path = Path("golden_set/golden_set.jsonl")
output_path = Path("golden_set/corrections/corrections.csv")

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def clean_boxes(boxes):
    keys = ("left", "top", "width", "height")
    return [{key: box[key] for key in keys} for box in boxes]

def canonical(boxes):
    return sorted(
        tuple(round(float(box[key]), 12)
              for key in ("left", "top", "width", "height"))
        for box in boxes
    )

reviews = read_jsonl(review_path)
golden = {
    str(row["photo_id"]): row["boxes"]
    for row in read_jsonl(golden_path)
}

assert len(reviews) == 400
assert len(golden) == 400

corrections = []

for row in reviews:
    photo_id = str(row["input_values"]["photo_id"])
    original = clean_boxes(row["input_values"]["boxes"])
    corrected = clean_boxes(row["output_values"]["boxes"])

    assert canonical(corrected) == canonical(golden[photo_id])

    if row["output_values"]["review_verdict"] != "reject":
        continue

    if len(original) < len(corrected):
        change = "added_box"
        explanation = "Ревьюер добавил пропущенную рамку"
    elif len(original) > len(corrected):
        change = "removed_box"
        explanation = "Ревьюер удалил исходную рамку"
    else:
        change = "adjusted_boxes"
        explanation = "Ревьюер скорректировал координаты рамок"

    corrections.append({
        "photo_id": photo_id,
        "change_type": change,
        "explanation": explanation,
        "original_boxes": json.dumps(original),
        "corrected_boxes": json.dumps(corrected),
        "reviewer_comment": row["output_values"].get("comment", ""),
        "team_decision": "approved_without_additional_changes",
    })

assert len(corrections) == 13
corrections.sort(key=lambda row: int(row["photo_id"]))

with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(corrections[0].keys())
    )
    writer.writeheader()
    writer.writerows(corrections)

print("Всего исправлений:", len(corrections))
print("Типы изменений:", dict(Counter(
    row["change_type"] for row in corrections
)))
print("Файл:", output_path)
