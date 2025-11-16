import argparse
import os
import sys
import time
from typing import Dict, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# -------------------------------
# Utilities
# -------------------------------

def safe_color(bgr_tuple):
    return tuple(int(max(0, min(255, c))) for c in bgr_tuple)


MODEL_COLORS = {
    "markings": safe_color((0, 255, 0)),      # Green
    "pothole": safe_color((0, 0, 255)),       # Red
    "sign": safe_color((255, 0, 0)),          # Blue
    "vulnerable": safe_color((255, 255, 0)),  # Cyan/Yellow
}


# -------------------------------
# Model Loading
# -------------------------------

def load_models(markings_path: str, pothole_path: str, sign_path: str, vulnerable_path: str,
                device: str = "auto") -> Dict[str, Any]:
    """Load YOLOv8 models and a YOLOv5 model.

    Returns a dict of loaded models; if a path is missing, that model is skipped.
    """
    models: Dict[str, Any] = {}

    def _exists(p):
        return p and os.path.exists(p)

    # YOLOv8 models via ultralytics
    if _exists(markings_path):
        try:
            print(f"Loading YOLOv8 markings model from {markings_path} ...")
            models["markings"] = YOLO(markings_path)
        except Exception as e:
            print(f"Failed to load markings model: {e}")

    if _exists(pothole_path):
        try:
            print(f"Loading YOLOv8 pothole model from {pothole_path} ...")
            models["pothole"] = YOLO(pothole_path)
        except Exception as e:
            print(f"Failed to load pothole model: {e}")

    if _exists(sign_path):
        try:
            print(f"Loading YOLOv8 sign model from {sign_path} ...")
            models["sign"] = YOLO(sign_path)
        except Exception as e:
            print(f"Failed to load sign model: {e}")

    # YOLOv5 model via torch.hub (requires internet on first use)
    if _exists(vulnerable_path):
        try:
            print(f"Loading YOLOv5 vulnerable road users model from {vulnerable_path} ...")
            # This downloads ultralytics/yolov5 repo if not present
            models["vulnerable"] = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=vulnerable_path, force_reload=False
            )
            if device != "cpu" and torch.cuda.is_available():
                models["vulnerable"].to('cuda')
        except Exception as e:
            print(f"Failed to load vulnerable model: {e}")

    if not models:
        print("No models loaded. Please check the paths.")

    return models


# -------------------------------
# Inference per frame
# -------------------------------

def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def process_with_yolov8(model: YOLO, frame: np.ndarray, color, conf_thres: float):
    results = model(frame, conf=conf_thres, verbose=False)
    annot = frame
    res = results[0]
    if res.boxes is None or res.boxes.xyxy is None:
        return annot

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.zeros((xyxy.shape[0],))
    clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)

    names = model.names if hasattr(model, 'names') else {}

    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = map(int, xyxy[i, :4])
        conf = float(confs[i]) if i < len(confs) else 0.0
        cls_id = int(clss[i]) if i < len(clss) else -1
        label = names.get(cls_id, f"cls{cls_id}") if isinstance(names, dict) else str(cls_id)
        cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
        draw_label(annot, f"{label} {conf:.2f}", x1, y1, color)
    return annot


def process_with_yolov5(model, frame: np.ndarray, color, conf_thres: float):
    # model is a YOLOv5 hub model
    model.conf = conf_thres  # set confidence threshold
    results = model(frame)   # inference
    if not hasattr(results, 'xyxy') or len(results.xyxy) == 0:
        return frame

    annot = frame
    dets = results.xyxy[0].cpu().numpy()
    names = model.names if hasattr(model, 'names') else {}

    for *xyxy, conf, cls_id in dets:
        if float(conf) < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        label = names[int(cls_id)] if isinstance(names, (list, dict)) and int(cls_id) in names else f"cls{int(cls_id)}"
        cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
        draw_label(annot, f"{label} {float(conf):.2f}", x1, y1, color)
    return annot


# -------------------------------
# Main video loop
# -------------------------------

def run_video(models: Dict[str, Any], input_path: str, output_path: str, conf: float, show: bool):
    if not os.path.exists(input_path):
        print(f"Input video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nProcessing video...\n - Input: {input_path}\n - Output: {output_path}\n - FPS: {fps:.2f}, Size: {width}x{height}\n - Models: {', '.join(models.keys()) or 'None'}\n")

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame
            # Process with each model
            for name, model in models.items():
                color = MODEL_COLORS.get(name, (255, 255, 255))
                if name == "vulnerable":
                    annotated = process_with_yolov5(model, annotated, color, conf)
                else:
                    annotated = process_with_yolov8(model, annotated, color, conf)

            # Composite label to indicate which colors map to which models
            y = 24
            for name, color in MODEL_COLORS.items():
                if name in models:
                    cv2.putText(annotated, f"{name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    y += 22

            out.write(annotated)
            if show:
                cv2.imshow('Detections (press q to quit)', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user.")
                    break

            frame_idx += 1
            if frame_idx % 25 == 0:
                elapsed = time.time() - t0
                print(f"Processed {frame_idx} frames, {frame_idx / max(1.0, elapsed):.2f} FPS")

    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    print("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple YOLO models on a video and save visualized output.")
    p.add_argument('--input', '-i', type=str, required=True, help='Path to input video file')
    p.add_argument('--output', '-o', type=str, default='output_detections.mp4', help='Path to output video file')
    p.add_argument('--markings', type=str, default='best_markings.pt', help='Path to markings YOLOv8 weights')
    p.add_argument('--pothole', type=str, default='best_pothole.pt', help='Path to pothole YOLOv8 weights')
    p.add_argument('--sign', type=str, default='best_sign.pt', help='Path to sign YOLOv8 weights')
    p.add_argument('--vulnerable', type=str, default='vulnerable_best.pt', help='Path to vulnerable users YOLOv5 weights')
    p.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for all models')
    p.add_argument('--show', action='store_true', help='Show live window while processing')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device preference')
    return p.parse_args()


def main():
    args = parse_args()

    # Memory/GPU housekeeping
    torch.cuda.empty_cache()

    # Device info
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")

    models = load_models(args.markings, args.pothole, args.sign, args.vulnerable, device=args.device)
    if not models:
        sys.exit(1)

    run_video(models, args.input, args.output, conf=args.conf, show=args.show)


if __name__ == '__main__':
    main()
