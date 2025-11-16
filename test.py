from ultralytics import YOLO
model = YOLO("best_sign.pt")

# Run detection on a road video
results = model.predict(
    source="speedsign.png",  # your image path
    conf=0.5,                # confidence threshold
    save=True,                # save annotated image
    show=True                 # display annotated image
)

# Optional: print detection info
for r in results:
    boxes = r.boxes.xyxy        # x1, y1, x2, y2
    classes = r.boxes.cls       # class indices
    confs = r.boxes.conf        # confidence scores
    names = [model.names[int(i)] for i in classes]
    print(list(zip(names, confs.tolist())))