"""
Real-time video detection using Roboflow model downloaded locally
This downloads the model once and then runs it locally for real-time processing
"""
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import torch
import os
import time

# Clear GPU cache for better performance
torch.cuda.empty_cache()

# Initialize Roboflow and download model
rf = Roboflow(api_key="Bj727w9uyjQFBN2OgnjK")
project = rf.workspace().project("my-first-project-luxe7")
model_version = project.version(2)

# Download model weights to local file
print("📥 Downloading Roboflow model locally...")
dataset = model_version.download("yolov8")

# Get the directory path - Roboflow typically downloads to "ProjectName-Version"
# Based on the download output, it should be in the current directory
current_dir = os.getcwd()
model_dir = None

# Check for common Roboflow download directory patterns
possible_dirs = [
    "My-First-Project-2",  # Based on terminal output
    f"my-first-project-luxe7-{model_version.id}",
    f"My-First-Project-{model_version.id}",
]

# Also check if dataset has a location attribute
if hasattr(dataset, 'location'):
    model_dir = dataset.location
elif hasattr(dataset, 'path'):
    model_dir = dataset.path
else:
    # Search for the downloaded directory
    for possible_dir in possible_dirs:
        check_path = os.path.join(current_dir, possible_dir)
        if os.path.exists(check_path):
            model_dir = check_path
            break

# If still not found, search for directories with .pt files
if model_dir is None:
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and ("project" in item.lower() or "first" in item.lower()):
            # Check if it contains weights
            weights_check = os.path.join(item_path, "weights", "best.pt")
            if os.path.exists(weights_check):
                model_dir = item_path
                break

if model_dir is None:
    print(f"❌ Error: Could not locate downloaded model directory")
    print(f"   Current directory: {current_dir}")
    print(f"   Tried: {possible_dirs}")
    exit()

print(f"✅ Model downloaded to: {model_dir}")

# Find the model weights file
weights_path = os.path.join(model_dir, "weights", "best.pt")
if not os.path.exists(weights_path):
    # Sometimes the structure might be different, try alternatives
    alt_path = os.path.join(model_dir, "best.pt")
    if os.path.exists(alt_path):
        weights_path = alt_path
    else:
        print(f"❌ Error: Could not find model weights file")
        print(f"   Checked: {weights_path}")
        print(f"   Checked: {alt_path}")
        print(f"   Available files in {model_dir}:")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.pt'):
                    print(f"      {os.path.join(root, file)}")
        exit()

# Load the downloaded model
print(f"🔄 Loading model from: {weights_path}")
model = YOLO(weights_path)

# Configuration
input_video_path = "Trial_1_Video.MP4"  # Change to your video path
output_video_path = "roboflow_realtime_detection.mp4"
confidence_threshold = 0.5
show_preview = True  # Set to False to disable live preview (faster processing)

# Check if video file exists
if not os.path.exists(input_video_path):
    print(f"❌ Error: Video file not found: {input_video_path}")
    print(f"Available videos: Trial_1_Video.MP4, Trial_2_Video.mp4")
    exit()

# Open the video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"❌ Error: Could not open video file: {input_video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n📹 Video Info:")
print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps}")
print(f"   Total frames: {total_frames}")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Processing stats
frame_count = 0
start_time = time.time()
fps_list = []

print(f"\n🔄 Processing video with real-time detection (Roboflow model)...")
print(f"   Press 'q' to quit preview\n")

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_start = time.time()
    frame_count += 1
    
    # Run model on frame (disable gradient computation for speed)
    with torch.no_grad():
        results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
    
    # Draw detections
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = r.names[int(cls)]
            confidence = conf.item()
            
            # Draw bounding box (green)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                          (x1 + label_size[0], label_y + 5), color, -1)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Calculate processing FPS
    frame_time = time.time() - frame_start
    current_fps = 1.0 / frame_time if frame_time > 0 else 0
    fps_list.append(current_fps)
    
    # Add FPS counter to frame
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Write processed frame to output video
    out.write(frame)
    
    # Show progress every 30 frames
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        avg_fps = sum(fps_list[-30:]) / min(30, len(fps_list))
        print(f"   Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Avg FPS: {avg_fps:.1f}")
    
    # Display live preview
    if show_preview:
        cv2.imshow("Real-time Detection (Roboflow Model)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ Processing stopped by user")
            break

# Cleanup
cap.release()
out.release()
if show_preview:
    cv2.destroyAllWindows()

# Final stats
total_time = time.time() - start_time
avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

print(f"\n✅ Video processing completed!")
print(f"   Input: {input_video_path}")
print(f"   Output: {output_video_path}")
print(f"   Total frames: {frame_count}")
print(f"   Processing time: {total_time:.2f}s")
print(f"   Average FPS: {avg_fps:.2f}")
print(f"   Speed: {total_time / frame_count * fps:.2f}x real-time")

