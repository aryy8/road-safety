import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
present_csv = "detections.csv"
base_csv = "detections_base.csv"       # optional
out_folder = "audit_report_outputs"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# ------------------------------------------------------------
# LOAD PRESENT DATA
# ------------------------------------------------------------
print("Loading present detections...")
df = pd.read_csv(present_csv)

has_base = os.path.exists(base_csv)
if has_base:
    print("Loading base detections...")
    df_base = pd.read_csv(base_csv)

# ------------------------------------------------------------
# HELPER: SAFE PLOT SAVE
# ------------------------------------------------------------
def save_plot(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ------------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ------------------------------------------------------------
class_dist_png = os.path.join(out_folder, "class_distribution.png")

plt.figure(figsize=(10,5))
df["class"].value_counts().plot(kind="bar")
plt.title("Detections per Class (Present Video)")
plt.xlabel("Class")
plt.ylabel("Count")
save_plot(class_dist_png)

# ------------------------------------------------------------
# 2. TREND PER FRAME
# ------------------------------------------------------------
trend_png = os.path.join(out_folder, "trend_per_frame.png")
frame_counts = df.groupby(["frame","class"]).size().unstack().fillna(0)

plt.figure(figsize=(14,6))
frame_counts.plot()
plt.title("Count per Class Across Frames")
plt.xlabel("Frame")
plt.ylabel("Count")
save_plot(trend_png)

# ------------------------------------------------------------
# 3. AREA TRENDS FOR KEY ELEMENTS
# ------------------------------------------------------------
important = ["road-marking", "edgeline", "divider"]
trend_imgs = []

for c in important:
    sub = df[df["class"] == c]
    if sub.empty:
        continue
    png = os.path.join(out_folder, f"{c}_area_trend.png")
    area = sub.groupby("frame")["bbox_area"].mean()

    plt.figure(figsize=(14,5))
    area.plot()
    plt.title(f"Average Bounding Box Area Trend: {c}")
    plt.xlabel("Frame")
    plt.ylabel("Area")
    save_plot(png)

    trend_imgs.append(png)

# ------------------------------------------------------------
# 4. MISSING ELEMENT DETECTION
# ------------------------------------------------------------
missing_rows = []
for c in df["class"].unique():
    per_frame = df[df["class"] == c].groupby("frame").size()
    zeros = per_frame[per_frame == 0].index.tolist()
    if not zeros:
        continue
    missing_rows.append({"class": c, "missing_frames": zeros[:20]})

missing_df = pd.DataFrame(missing_rows)
missing_csv = os.path.join(out_folder, "missing_elements.csv")
missing_df.to_csv(missing_csv, index=False)

# ------------------------------------------------------------
# 5. FADING DETECTION
# ------------------------------------------------------------
fade_rows = []

for c in important:
    sub = df[df["class"] == c]
    if sub.empty:
        continue
    area = sub.groupby("frame")["bbox_area"].mean()
    if len(area) < 10:
        continue
    drop = (area.iloc[0] - area.iloc[-1]) / max(area.iloc[0], 1)
    fade_rows.append({"class": c, "fading_ratio": round(drop*100, 2)})

fade_df = pd.DataFrame(fade_rows)
fade_csv = os.path.join(out_folder, "fading_summary.csv")
fade_df.to_csv(fade_csv, index=False)

# ------------------------------------------------------------
# 6. BASE VS PRESENT COMPARISON (if base exists)
# ------------------------------------------------------------
diff_rows = []
if has_base:
    base_counts = df_base["class"].value_counts()
    present_counts = df["class"].value_counts()

    all_classes = set(base_counts.index).union(set(present_counts.index))

    for c in all_classes:
        b = base_counts.get(c, 0)
        p = present_counts.get(c, 0)
        delta = p - b
        diff_rows.append({"class": c, "base": b, "present": p, "difference": delta})

diff_df = pd.DataFrame(diff_rows)
diff_csv = os.path.join(out_folder, "base_vs_present.csv")
diff_df.to_csv(diff_csv, index=False)

# ------------------------------------------------------------
# 7. ROAD RISK SCORING
# ------------------------------------------------------------
risk_rows = []

for c in df["class"].unique():
    sub = df[df["class"] == c]
    per_frame = sub.groupby("frame").size()
    var = per_frame.std()
    area_var = sub["bbox_area"].std()

    fade_score = 0
    if c in fade_df["class"].values:
        fade_score = float(fade_df[fade_df["class"] == c]["fading_ratio"].iloc[0])

    risk = fade_score + (var * 0.5) + (area_var * 0.1)
    risk_rows.append({"class": c, "risk_score": round(risk, 2)})

risk_df = pd.DataFrame(risk_rows)
risk_csv = os.path.join(out_folder, "risk_scores.csv")
risk_df.to_csv(risk_csv, index=False)

# ------------------------------------------------------------
# 8. CREATE PDF REPORT
# ------------------------------------------------------------
pdf_path = os.path.join(out_folder, "road_safety_report.pdf")
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

def add_title(text):
    pdf.set_font("Arial", size=18)
    pdf.multi_cell(0, 10, text)
    pdf.ln(3)

def add_text(text):
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, text)
    pdf.ln(2)

def add_img(path, w=170):
    pdf.image(path, w=w)
    pdf.ln(5)

pdf.add_page()
add_title("Road Safety Automated Audit Report")
add_text("Generated using road infrastructure detections and analytical methods.")

# Summary
pdf.set_font("Arial", size=14)
pdf.ln(3)
add_title("Analysis Overview")
add_text("This report includes missing element detection, fading analysis, base vs present comparison, and risk scoring.")

# Plots
add_title("Class Distribution")
add_img(class_dist_png)

add_title("Trend per Frame")
add_img(trend_png)

for img in trend_imgs:
    title = img.split("/")[-1].replace("_", " ").replace(".png", "")
    add_title(title)
    add_img(img)

# Missing
add_title("Missing Element Detection")
if missing_df.empty:
    add_text("No missing elements detected.")
else:
    for _, row in missing_df.iterrows():
        add_text(f"{row['class']} missing at frames: {row['missing_frames']}")

# Fading
add_title("Fading Analysis")
if fade_df.empty:
    add_text("No fading elements detected.")
else:
    for _, row in fade_df.iterrows():
        add_text(f"{row['class']} fading ratio: {row['fading_ratio']} percent")

# Base vs present
if has_base:
    add_title("Base vs Present Comparison")
    for _, row in diff_df.iterrows():
        add_text(f"{row['class']}: base {row['base']}, present {row['present']}, change {row['difference']}")

# Risk scores
add_title("Road Risk Scores")
for _, row in risk_df.iterrows():
    add_text(f"{row['class']} risk score: {row['risk_score']}")

pdf.output(pdf_path)

print("\nPDF saved at:", pdf_path)
print("All outputs stored in:", out_folder)
