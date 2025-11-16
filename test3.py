from ultralytics import YOLO
import cv2
import torch

# Clear any leftover GPU memory
torch.cuda.empty_cache()

# Load the trained road markings YOLO model
model = YOLO("best_markings.pt")

# Input and output paths
input_video_path = "Trial_1_Video.MP4"     # Replace with your video
output_video_path = "markings_detect.mp4"

# Open the video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Inference on one frame (disable gradients for efficiency)
    with torch.no_grad():
        results = model.predict(source=frame, conf=0.5, verbose=False)

    # Draw detections
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]  # Class name
            confidence = conf.item()

            # Assign each marking class a unique color (optional)
            color = (255, 255, 0)  # Yellow for road markings
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Write processed frame to output video
    out.write(frame)

    # Optional: show live window
    cv2.imshow("Road Markings Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Detection completed! Video saved as '{output_video_path}'")
