import time
from pathlib import Path
from ultralytics import YOLO

#---Configurations----
DATASET_ROOT= Path("datasets/anti_uav/train/20190925_101846_1_1")

VIDOES ={
    "visible": DATASET_ROOT / "visible.mp4",
    "thermal": DATASET_ROOT / "infrared.mp4"
}

MODELS={
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
}

OUTPUT_ROOT= Path("outputs/benchmark_yolo")
LOG_FILE= OUTPUT_ROOT / "benchmark_log.txt"

# Lowering the image size for cpu
IMG_SIZE= 640

if LOG_FILE.exists():
    LOG_FILE.unlink()

#--- Benchmark Functions---
def run_benchmark(model_name, model_weights, video_name, video_path):
   
    print(f"Running benchmark for Model: {model_name}, Video: {video_name}")
    model = YOLO(model_weights)
    
    start= time.time()

    results= model(str(video_path), save=True, imgsz=IMG_SIZE, project=str(OUTPUT_ROOT), name=f"{model_name}_{video_name}", exist_ok=True, verbose=True)
    
    

    """
    1. Look up the results section and variable used for documentation 
    2. What is the use of time library?
    3. Why are we using exist_ok and verobose? What exactly do they do?
    
    """

    end= time.time()
    log_text= (
        f"model:{model_name}\n"
        f"video:{video_name}\n"
        f"path:{video_path}\n"
        f"total_time_sec:{end - start:.2f}\n"
        f"{'-'*50}\n" 
    )

    """
    1. Why are we using f"{'-'*50}\n" 
    """

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a") as f:
        f.write(log_text)
    
    print(log_text)
    del model # What is the use of del keyword here?

#--- Main Execution---
if __name__ == "__main__":
    for model_name, model_weights in MODELS.items():
        for video_name, video_path in VIDOES.items():
            if video_path.exists():
                run_benchmark(model_name, model_weights, video_name, video_path)

    # Handling missing files case so that it doesnt crash the entire benchmark
        else:
            print(f"Skipping missing file: {video_path}")


"""
imgsz=640
imgsz=512
imgsz=416
imgsz=320
imgsz=224

What is the use of imgsz parameter? What does it do? Why are we using it here?

"""