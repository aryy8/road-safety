import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os

# ------------------------------------------------------------
# LOAD CSV
# ------------------------------------------------------------

csv_path = "detections.csv"
df = pd.read_csv(csv_path)
print("Loaded detections.csv with", len(df), "rows")

# ------------------------------------------------------------
# MAKE OUTPUT FOLDER
# ------------------------------------------------------------
if not os.path.exists("report_outputs"):
    os.makedirs("report_outputs")

# ------------------------------------------------------------
# 1. CLASS DISTRIBUTION PLOT
# ------------------------------------------------------------
class_png = "report_outputs/class_distribution.png"

plt.figure(figsize=(8,5))
df["class"].value_counts().plot(kind="bar")
plt.title("Detections per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(class_png)
plt.close()

# ------------------------------------------------------------
# 2. COUNT PER FRAME TREND
# ------------------------------------------------------------
trend_png = "report_outputs/trend_per_frame.png"

frame_counts = df.groupby(["frame","class"]).size().unstack().fillna(0)

plt.figure(figsize=(12,6))
frame_counts.plot()
plt.title("Count per Class across Frames")
plt.xlabel("Frame")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(trend_png)
plt.close()

# ------------------------------------------------------------
# 3. AREA TREND FOR IMPORTANT CLASSES
# ------------------------------------------------------------
important_classes = ["road-marking", "edgeline", "divider"]
area_plots = []

for c in important_classes:
    class_df = df[df["class"] == c]
    if class_df.empty:
        continue
    area_png = f"report_outputs/{c}_area_trend.png"
    avg_area = class_df.groupby("frame")["bbox_area"].mean()

    plt.figure(figsize=(12,5))
    avg_area.plot()
    plt.title(f"Average Area Trend for {c}")
    plt.xlabel("Frame")
    plt.ylabel("Average BBox Area")
    plt.tight_layout()
    plt.savefig(area_png)
    plt.close()

    area_plots.append(area_png)

# ------------------------------------------------------------
# 4. SUMMARY TABLE EXPORT
# ------------------------------------------------------------
summary_rows = []

for c in df["class"].unique():
    d = df[df["class"] == c]
    g = d.groupby("frame").size()

    summary_rows.append({
        "class": c,
        "avg_count": round(g.mean(), 2),
        "count_variation": round(g.std(), 2),
        "avg_bbox_area": round(d["bbox_area"].mean(), 2),
        "area_variation": round(d["bbox_area"].std(), 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = "report_outputs/audit_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print("Saved:", summary_csv)

# ------------------------------------------------------------
# 5. GENERATE PDF REPORT
# ------------------------------------------------------------

pdf_path = "report_outputs/road_safety_report.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Road Safety Automated Audit Report", styles['Title']))
story.append(Spacer(1, 20))
story.append(Paragraph("This PDF contains the analysis generated from your YOLO road infrastructure detections.", styles['BodyText']))
story.append(Spacer(1, 20))

# Add plots
story.append(Paragraph("Detections per Class", styles['Heading2']))
story.append(Image(class_png, width=5*inch, height=3*inch))
story.append(Spacer(1, 20))

story.append(Paragraph("Count Trend per Frame", styles['Heading2']))
story.append(Image(trend_png, width=5*inch, height=3*inch))
story.append(Spacer(1, 20))

for img in area_plots:
    story.append(Paragraph(os.path.basename(img).replace("_"," ").title(), styles['Heading2']))
    story.append(Image(img, width=5*inch, height=3*inch))
    story.append(Spacer(1, 20))

# Add summary table text
story.append(Paragraph("Summary of Detections", styles['Heading2']))
for _, row in summary_df.iterrows():
    story.append(Paragraph(
        f"{row['class']}: avg_count={row['avg_count']}, "
        f"count_variation={row['count_variation']}, "
        f"avg_bbox_area={row['avg_bbox_area']}, "
        f"area_variation={row['area_variation']}",
        styles['BodyText']
    ))
    story.append(Spacer(1, 10))

doc.build(story)
print("PDF saved:", pdf_path)
