from ultralytics import YOLO
import cv2
import torch

# Clear any cached GPU memory
torch.cuda.empty_cache()

# Load your three trained YOLO models
sign_model = YOLO("best_sign.pt")
pothole_model = YOLO("best_pothole.pt")
markings_model = YOLO("best_markings.pt")

# Input image path
image_path = "Screenshot 2025-10-27 at 9.41.19 PM.png"

# Read the image
frame = cv2.imread(image_path)

# Ensure image loaded
if frame is None:
    print("❌ Error: Could not read the image.")
    exit()

# Run all models on the same image (disable gradients for efficiency)
with torch.no_grad():
    sign_results = sign_model.predict(source=frame, conf=0.5, verbose=False)
    pothole_results = pothole_model.predict(source=frame, conf=0.5, verbose=False)
    markings_results = markings_model.predict(source=frame, conf=0.5, verbose=False)

# Draw road sign detections (🟩 Green)
for r in sign_results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"Sign {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Draw pothole detections (🟥 Red)
for r in pothole_results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"Pothole {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Draw road marking detections (🟦 Blue)
for r in markings_results:
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"Marking {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display combined detections
cv2.imshow("Combined Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the combined output image
output_path = "combined_detection.png"
cv2.imwrite(output_path, frame)
print(f"✅ Combined detection saved as '{output_path}'")
