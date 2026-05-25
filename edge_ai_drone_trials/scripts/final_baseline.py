import csv
import gc
import torch
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CPU SAFETY
# ==========================================
torch.set_num_threads(4)

# ==========================================
# CONFIG
# ==========================================
DATASET_ROOT = Path("datasets/anti_uav/train/20190925_101846_1_1")

VIDEOS = {
    "visible": DATASET_ROOT / "visible.mp4",
    "infrared": DATASET_ROOT / "infrared.mp4"
}

MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt"
}

IMG_SIZES = [640, 512, 416, 320, 224]

OUTPUT_ROOT = Path("outputs/final_baseline")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CSV_LOG = OUTPUT_ROOT / "baseline_metrics.csv"

# ==========================================
# CSV INIT
# ==========================================
def initialize_csv():
    with open(CSV_LOG, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "video",
            "imgsz",
            "frame_count",
            "pure_pipeline_time_sec",
            "fps"
        ])

# ==========================================
# BENCHMARK
# ==========================================
def run_benchmark(model_name, model_weights, video_name, video_path, img_size):
    print(f"\nRunning {model_name} | {video_name} | imgsz={img_size}")

    model = YOLO(model_weights)

    results_generator = model(
        source=str(video_path),
        save=False,
        imgsz=img_size,
        stream=True,
        device="cpu",
        verbose=False
    )

    frame_count = 0
    pure_pipeline_time = 0.0

    for result in results_generator:
        frame_count += 1

        frame_time_ms = (
            result.speed["preprocess"]
            + result.speed["inference"]
            + result.speed["postprocess"]
        )

        pure_pipeline_time += frame_time_ms / 1000.0

    fps = frame_count / pure_pipeline_time if pure_pipeline_time > 0 else 0

    with open(CSV_LOG, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            video_name,
            img_size,
            frame_count,
            round(pure_pipeline_time, 3),
            round(fps, 2)
        ])

    print(
        f"Completed → Frames: {frame_count} | "
        f"Time: {pure_pipeline_time:.2f}s | "
        f"FPS: {fps:.2f}"
    )

    del model
    gc.collect()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    initialize_csv()

    for model_name, model_weights in MODELS.items():
        for video_name, video_path in VIDEOS.items():
            if not video_path.exists():
                print(f"Skipping missing file: {video_path}")
                continue

            for img_size in IMG_SIZES:
                run_benchmark(
                    model_name,
                    model_weights,
                    video_name,
                    video_path,
                    img_size
                )