from ultralytics import YOLO
import cv2
import torch

# Clear GPU cache before starting
torch.cuda.empty_cache()

# Load your trained YOLO models
# sign_model = YOLO("best_sign.pt")
# pothole_model = YOLO("best_pothole.pt")
markings_model = YOLO("best_markings.pt")

# Input and output paths
input_video_path = "Trial_2_Video.MP4"     # Replace with your video file
output_video_path = "combined_detection.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video details
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run all three models on the same frame (disable gradient computation)
    with torch.no_grad():
        # sign_results = sign_model.predict(source=frame, conf=0.5, verbose=False)
        # pothole_results = pothole_model.predict(source=frame, conf=0.5, verbose=False)
        marking_results = markings_model.predict(source=frame, conf=0.5, verbose=False)

    # Draw road sign detections (Green)
    # for r in sign_results:
    #     for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
    #         x1, y1, x2, y2 = map(int, box)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, f"Sign {conf:.2f}", (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # # Draw pothole detections (Red)
    # for r in pothole_results:
    #     for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
    #         x1, y1, x2, y2 = map(int, box)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         cv2.putText(frame, f"Pothole {conf:.2f}", (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw road markings detections (Blue)
    for r in marking_results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Marking {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Write the combined frame
    out.write(frame)

    # (Optional) Display live output
    cv2.imshow("Combined Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Combined detection video saved as '{output_video_path}'")
