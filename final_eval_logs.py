from inference import InferencePipeline
import cv2
import csv
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
CSV_PATH = "detections.csv"
VIDEO_PATH = "/Users/aryan/road_safety/Trial_1_Video.MP4"

# ------------------------------------------------------------
# CREATE CSV WRITER
# ------------------------------------------------------------
write_header = not os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)

if write_header:
    csv_writer.writerow([
        "frame",
        "class",
        "confidence",
        "x1", "y1", "x2", "y2",
        "width",
        "height",
        "bbox_area",
        "uuid"
    ])

# ------------------------------------------------------------
# GLOBAL FRAME COUNTER
# ------------------------------------------------------------
frame_number = 0

# ------------------------------------------------------------
# CALLBACK TO PROCESS EACH FRAME
# ------------------------------------------------------------
def my_sink(result, video_frame):
    global frame_number, csv_writer

    preds = result.get("predictions", None)

    # If no detections
    if preds is None or len(preds.xyxy) == 0:
        frame_number += 1
        return

    xyxy = preds.xyxy
    class_names = preds.data["class_name"]
    confs = preds.confidence
    ids = preds.data["detection_id"]

    # Write each detection to CSV
    for i in range(len(class_names)):
        x1, y1, x2, y2 = xyxy[i]

        csv_writer.writerow([
            frame_number,
            class_names[i],
            float(confs[i]),
            int(x1), int(y1), int(x2), int(y2),
            int(x2 - x1),
            int(y2 - y1),
            int((x2 - x1) * (y2 - y1)),
            ids[i]
        ])

    frame_number += 1

    # Optional live preview (safe)
    if result.get("output_image"):
        cv2.imshow("Detections", result["output_image"].numpy_image)
        cv2.waitKey(1)

# ------------------------------------------------------------
# RUN PIPELINE
# ------------------------------------------------------------
pipeline = InferencePipeline.init_with_workflow(
    api_key="Bj727w9uyjQFBN2OgnjK",
    workspace_name="aryy8",
    workflow_id="detect-count-and-visualize",
    video_reference=VIDEO_PATH,
    max_fps=30,
    on_prediction=my_sink
)

print("Processing video...")
pipeline.start()
pipeline.join()
cv2.destroyAllWindows()
csv_file.close()

print("\nAll detections saved to:", CSV_PATH)
