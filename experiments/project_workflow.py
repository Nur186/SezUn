"""Prepare COCO128, run EDA, and compare a YOLO baseline with two fine-tuned models.

Run from the repository root:
    python experiments/project_workflow.py --prepare-only
    python experiments/project_workflow.py --epochs 10

The dataset is downloaded to Ultralytics' external cache. Processed splits and
reports are written under reports/ and are excluded from source control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = REPORTS / "data" / "coco128"
SEED = 42
IMAGE_SIZE = 416
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def clean_label_file(source: Path, target: Path, class_count: int) -> tuple[int, int, int]:
    """Keep only finite, in-range YOLO boxes; report missing/malformed rows."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        target.write_text("", encoding="utf-8")
        return 0, 1, 0
    valid_lines: list[str] = []
    invalid = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError
            class_id = int(parts[0])
            box = [float(value) for value in parts[1:]]
            if class_id < 0 or class_id >= class_count or not all(math.isfinite(x) for x in box):
                raise ValueError
            if any(x < 0 or x > 1 for x in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError
        except ValueError:
            invalid += 1
            continue
        valid_lines.append(" ".join([str(class_id), *(f"{x:.6f}" for x in box)]))
    target.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
    return len(valid_lines), 0, invalid


def prepare_dataset() -> tuple[Path, dict]:
    info = check_det_dataset("coco128.yaml")
    names = info["names"]
    class_count = len(names)
    dataset_root = Path(info["path"])
    source_images = Path(info["train"])
    if not source_images.is_absolute():
        source_images = dataset_root / source_images
    source_images = source_images.resolve()
    images = sorted(path for path in source_images.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    valid_images: list[Path] = []
    corrupt_images: list[str] = []
    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
            valid_images.append(path)
        except Exception:
            corrupt_images.append(path.name)
    if len(valid_images) < 20:
        raise RuntimeError(f"COCO128 жүктелмеді немесе кескін жеткіліксіз: {len(valid_images)}")

    rng = random.Random(SEED)
    rng.shuffle(valid_images)
    train_end = int(len(valid_images) * SPLIT_RATIOS["train"])
    val_end = train_end + int(len(valid_images) * SPLIT_RATIOS["val"])
    split_images = {"train": valid_images[:train_end], "val": valid_images[train_end:val_end], "test": valid_images[val_end:]}

    class_names = {int(key): str(value) for key, value in names.items()}
    class_counts: Counter[int] = Counter()
    box_areas: list[float] = []
    boxes_per_image: list[int] = []
    missing_labels = invalid_rows = total_boxes = 0
    split_counts: dict[str, dict] = {}
    for split, paths in split_images.items():
        image_out = PROCESSED / "images" / split
        label_out = PROCESSED / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        list_file = REPORTS / "data" / f"{split}.txt"
        out_paths: list[str] = []
        split_class_counts: Counter[int] = Counter()
        for source in paths:
            try:
                with Image.open(source) as image:
                    width, height = image.size
            except Exception:
                continue
            image_target = image_out / source.name
            shutil.copy2(source, image_target)
            label_source = label_path(source)
            label_target = label_out / f"{source.stem}.txt"
            box_count, missing, invalid = clean_label_file(label_source, label_target, class_count)
            missing_labels += missing
            invalid_rows += invalid
            total_boxes += box_count
            boxes_per_image.append(box_count)
            for row in label_target.read_text(encoding="utf-8").splitlines():
                class_text, cx, cy, box_w, box_h = row.split()
                class_id = int(class_text)
                class_counts[class_id] += 1
                split_class_counts[class_id] += 1
                box_areas.append(float(box_w) * float(box_h))
            out_paths.append(str(image_target.resolve()))
        list_file.write_text("\n".join(out_paths) + "\n", encoding="utf-8")
        split_counts[split] = {
            "images": len(out_paths),
            "boxes": sum(split_class_counts.values()),
            "classes_present": len(split_class_counts),
        }

    yaml_path = REPORTS / "data" / "coco128_split.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_lines = ["path: " + PROCESSED.as_posix(), "train: " + (REPORTS / "data" / "train.txt").as_posix(), "val: " + (REPORTS / "data" / "val.txt").as_posix(), "test: " + (REPORTS / "data" / "test.txt").as_posix(), "names:"]
    yaml_lines.extend(f"  {class_id}: {class_names[class_id]!r}" for class_id in sorted(class_names))
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary = {
        "dataset": "COCO128 (Ultralytics subset of COCO train2017)",
        "source": "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml",
        "license_note": "See the source dataset and COCO terms before redistribution.",
        "seed": SEED,
        "classes": class_names,
        "images_total": sum(item["images"] for item in split_counts.values()),
        "boxes_total": total_boxes,
        "missing_label_files": missing_labels,
        "corrupt_images_excluded": corrupt_images,
        "invalid_label_rows_removed": invalid_rows,
        "split": split_counts,
        "class_counts": {class_names[key]: value for key, value in sorted(class_counts.items())},
        "box_area_normalized_mean": sum(box_areas) / max(1, len(box_areas)),
        "empty_label_images": sum(1 for count in boxes_per_image if count == 0),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_eda_plots(summary, class_counts, class_names, box_areas, boxes_per_image)
    write_eda_report(summary)
    return yaml_path, summary


def write_eda_plots(summary: dict, counts: Counter, names: dict, areas: list[float], boxes_per_image: list[int]) -> None:
    figures = REPORTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    top = counts.most_common(20)
    plt.figure(figsize=(12, 6))
    plt.bar([names[key] for key, _ in top], [value for _, value in top], color="#287c72")
    plt.xticks(rotation=55, ha="right")
    plt.ylabel("Нысан саны")
    plt.title("COCO128: жиі кездесетін 20 сынып")
    plt.tight_layout()
    plt.savefig(figures / "class_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(areas, bins=24, color="#e29a45", edgecolor="white")
    plt.xlabel("Нысан ауданы / сурет ауданы (нормаланған)")
    plt.ylabel("Қорап саны")
    plt.title("Нысан өлшемдерінің үлестірімі")
    plt.tight_layout()
    plt.savefig(figures / "box_area_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(boxes_per_image, bins=range(0, max(boxes_per_image, default=1) + 2), color="#6583b3", edgecolor="white", align="left")
    plt.xlabel("Бір суреттегі қорап саны")
    plt.ylabel("Сурет саны")
    plt.title("Суреттегі белгіленген нысандар саны")
    plt.tight_layout()
    plt.savefig(figures / "boxes_per_image.png", dpi=160)
    plt.close()


def write_eda_report(summary: dict) -> None:
    top_classes = sorted(summary["class_counts"].items(), key=lambda item: item[1], reverse=True)[:10]
    top_text = ", ".join(f"{name}: {count}" for name, count in top_classes)
    report = [
        "# Деректерді дайындау және EDA",
        "",
        "- **Дерек:** COCO128 — COCO train2017 жиынынан алынған 128 суреттік шағын үлгі.",
        "- **Тапсырма:** объект детекциясы; әр нысанға класс пен YOLO форматындағы bounding box берілген.",
        f"- **Өңделген сурет саны:** {summary['images_total']}; **bounding box саны:** {summary['boxes_total']}.",
        f"- **BBox аннотациясы жоқ суреттер:** {summary['empty_label_images']}; **бастапқы label файлы табылмаған суреттер:** {summary['missing_label_files']}.",
        f"- **Оқылмайтын кескіндер алынып тасталды:** {len(summary['corrupt_images_excluded'])}; **қате annotation жолдары алынып тасталды:** {summary['invalid_label_rows_removed']}.",
        f"- **Split:** {json.dumps(summary['split'], ensure_ascii=False)} (seed={summary['seed']}).",
        f"- **Ең жиі кездесетін 10 сынып:** {top_text}.",
        "- Бастапқы label файлы табылмаған екі сурет өңделген жиында бос белгімен қалды; оларда нысан болмауы да, белгісі түсіп қалуы да мүмкін.",
        "- Класс саны тең емес, сондықтан аз кездесетін сыныптар бойынша метрика тұрақсыз болуы мүмкін.",
        "",
        "## Үлестірім графиктері",
        "",
        "- `figures/class_distribution.png` — ең жиі кездесетін 20 сынып.",
        "- `figures/box_area_distribution.png` — объект ауданының үлестірімі.",
        "- `figures/boxes_per_image.png` — бір суреттегі объект саны.",
        "",
        "COCO128 шағын болғандықтан, бұл EDA мен оқу экспериментіне ыңғайлы, бірақ үлкен ортаға жалпылауға жеткіліксіз.",
    ]
    (REPORTS / "EDA.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def iou(box_a, box_b) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def box_error_analysis(model: YOLO, test_list: Path, output_csv: Path) -> tuple[int, int, int]:
    tp_total = fp_total = fn_total = 0
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "tp", "fp", "fn", "false_positive_classes", "missed_classes"])
        writer.writeheader()
        for image_name in test_list.read_text(encoding="utf-8").splitlines():
            if not image_name:
                continue
            image_path = Path(image_name)
            gt_path = PROCESSED / "labels" / "test" / f"{image_path.stem}.txt"
            ground_truth = []
            if gt_path.exists():
                for line in gt_path.read_text(encoding="utf-8").splitlines():
                    class_id, cx, cy, width, height = line.split()
                    with Image.open(image_path) as image:
                        width_px, height_px = image.size
                    cx, cy, bw, bh = float(cx) * width_px, float(cy) * height_px, float(width) * width_px, float(height) * height_px
                    ground_truth.append((int(class_id), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), False))
            prediction = model.predict(str(image_path), imgsz=IMAGE_SIZE, conf=0.35, verbose=False)[0]
            preds = []
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    preds.append((int(box.cls[0].item()), tuple(float(v) for v in box.xyxy[0].tolist()), float(box.conf[0].item())))
            tp = fp = 0
            false_classes: list[str] = []
            for class_id, pred_box, _ in sorted(preds, key=lambda item: item[2], reverse=True):
                choices = [(iou(pred_box, gt_box), index) for index, (gt_class, gt_box, used) in enumerate(ground_truth) if gt_class == class_id and not used]
                best_iou, best_index = max(choices, default=(0.0, -1))
                if best_iou >= 0.5:
                    gt_class, gt_box, _ = ground_truth[best_index]
                    ground_truth[best_index] = (gt_class, gt_box, True)
                    tp += 1
                else:
                    fp += 1
                    false_classes.append(str(model.names[class_id]))
            missed = [str(model.names[class_id]) for class_id, _, used in ground_truth if not used]
            fn = len(missed)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            if fp or fn:
                writer.writerow({"image": image_path.name, "tp": tp, "fp": fp, "fn": fn, "false_positive_classes": ", ".join(false_classes), "missed_classes": ", ".join(missed)})
    return tp_total, fp_total, fn_total


def benchmark(data_yaml: Path, epochs: int, batch: int, device: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    test_list = REPORTS / "data" / "test.txt"
    candidates = [
        ("YOLOv8n pretrained baseline", "yolov8n.pt", False),
        ("YOLOv8n fine-tuned", "yolov8n.pt", True),
        ("YOLOv8s fine-tuned", "yolov8s.pt", True),
    ]
    rows = []
    error_totals = []
    for name, weights, fine_tune in candidates:
        model = YOLO(str(ROOT / weights) if (ROOT / weights).exists() else weights)
        if fine_tune:
            safe_name = name.lower().replace(" ", "_")
            model.train(
                data=str(data_yaml), epochs=epochs, imgsz=IMAGE_SIZE, batch=batch,
                workers=0, device=device, seed=SEED, deterministic=True,
                project=str(REPORTS / "runs"), name=safe_name, exist_ok=True,
                plots=True, patience=max(3, epochs // 2), verbose=False,
            )
            best = Path(model.trainer.save_dir) / "weights" / "best.pt"
            model = YOLO(str(best))
        val = model.val(
            data=str(data_yaml), split="test", imgsz=IMAGE_SIZE, batch=batch,
            workers=0, device=device, plots=True, project=str(REPORTS / "runs"),
            name=name.lower().replace(" ", "_") + "_test", exist_ok=True, verbose=False,
        )
        tp, fp, fn = box_error_analysis(model, test_list, REPORTS / f"errors_{name.lower().replace(' ', '_')}.csv")
        rows.append({
            "model": name,
            "precision": float(val.box.mp),
            "recall": float(val.box.mr),
            "mAP50": float(val.box.map50),
            "mAP50_95": float(val.box.map),
            "inference_ms_per_image": float(val.speed.get("inference", 0)),
            "test_TP_IoU50": tp,
            "test_FP_IoU50": fp,
            "test_FN_IoU50": fn,
        })
        error_totals.append((name, tp, fp, fn))
        print(f"{name}: P={val.box.mp:.3f} R={val.box.mr:.3f} mAP50={val.box.map50:.3f} mAP50-95={val.box.map:.3f}")

    csv_path = REPORTS / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Модельдерді салыстыру және қателерді талдау",
        "",
        f"Эксперимент: COCO128; seed={SEED}; input={IMAGE_SIZE}; fine-tune epochs={epochs}; test split — бөлек holdout.",
        "Көрсеткіштер: Precision, Recall, mAP50, mAP50–95 және CPU/GPU inference уақыты; қате мысалдары `errors_*.csv` ішінде.",
        "YOLO validation графиктері мен confusion matrix `runs/` ішіне сақталады.",
        "",
        "| Model | Precision | Recall | mAP50 | mAP50-95 | ms/image | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['model']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['mAP50']:.3f} | {row['mAP50_95']:.3f} | {row['inference_ms_per_image']:.1f} | {row['test_TP_IoU50']} | {row['test_FP_IoU50']} | {row['test_FN_IoU50']} |"
        for row in rows
    )
    lines += [
        "",
        "## Түсіндіру және шектеулер",
        "",
        "- Pretrained baseline пен fine-tuned нұсқаларын салыстыру жергілікті шағын деректің әсерін көрсетеді; бұл өндірістік дәлдікке кепіл емес.",
        "- COCO128 тесті небәрі аз суреттен тұрады; сирек кластар бойынша метрика тұрақсыз. Қорытындыны толық COCO немесе арнайы жиналған test set-пен қайта тексеру керек.",
        "- Headphones COCO класында жоқ, сондықтан бұл салыстыру ол затты бағаламайды.",
        "- FPS/кідіріс камера, процессор, GPU және браузерге тәуелді.",
        "",
        "Ұсынылатын нұсқаны `mAP50–95`, Recall, FP/FN және жылдамдықты бірге қарап таңдау керек; тек бір санға сүйенбеңіз.",
    ]
    (REPORTS / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Download, clean, split, and run EDA only")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs for each trained model")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    args = parser.parse_args()
    data_yaml, summary = prepare_dataset()
    print(f"EDA дайын: {summary['images_total']} сурет, {summary['boxes_total']} box; есеп: {REPORTS / 'EDA.md'}")
    if not args.prepare_only:
        benchmark(data_yaml, args.epochs, args.batch, args.device)
        print(f"Салыстыру нәтижесі: {REPORTS / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
