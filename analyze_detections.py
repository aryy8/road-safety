import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("detections.csv")

print("Loaded", len(df), "detections")

# ------------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ------------------------------------------------------------
class_counts = df["class"].value_counts()
print("\nClass frequency:")
print(class_counts)

plt.figure(figsize=(8,5))
class_counts.plot(kind="bar")
plt.title("Total detections per class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("class_distribution.png")
print("Saved: class_distribution.png")

# ------------------------------------------------------------
# 2. COUNT PER FRAME (TREND)
# ------------------------------------------------------------
frame_counts = df.groupby(["frame","class"]).size().unstack().fillna(0)

plt.figure(figsize=(12,6))
frame_counts.plot()
plt.title("Count of each class across frames")
plt.xlabel("Frame")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("trend_per_frame.png")
print("Saved: trend_per_frame.png")

# ------------------------------------------------------------
# 3. DETERIORATION: ROAD MARKINGS / EDGELINE
# ------------------------------------------------------------
def plot_area_trend(class_name):
    subset = df[df["class"] == class_name]
    if subset.empty:
        print(f"No detections for: {class_name}")
        return

    avg_area = subset.groupby("frame")["bbox_area"].mean()

    plt.figure(figsize=(12,5))
    avg_area.plot()
    plt.title(f"Average bbox area trend for {class_name}")
    plt.xlabel("Frame")
    plt.ylabel("Average area")
    plt.tight_layout()
    plt.savefig(f"{class_name}_area_trend.png")
    print(f"Saved: {class_name}_area_trend.png")

plot_area_trend("road-marking")
plot_area_trend("edgeline")
plot_area_trend("divider")

# ------------------------------------------------------------
# 4. DETERIORATION DETECTION LOGIC
# ------------------------------------------------------------
summary_rows = []

for c in df["class"].unique():
    subset = df[df["class"] == c]
    grouped = subset.groupby("frame").size()

    avg_count = grouped.mean()
    variability = grouped.std()

    avg_area = subset["bbox_area"].mean()
    area_var = subset["bbox_area"].std()

    summary_rows.append({
        "class": c,
        "avg_count_per_frame": round(avg_count, 2),
        "count_variation": round(variability, 2),
        "avg_area": round(avg_area, 2),
        "area_variation": round(area_var, 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("audit_summary.csv", index=False)
print("\nSaved: audit_summary.csv")

print("\nDeterioration summary:")
print(summary_df)

# ------------------------------------------------------------
# 5. MISSING ELEMENT DETECTION
# ------------------------------------------------------------
missing_log = []

for c in df["class"].unique():
    per_frame_counts = df[df["class"] == c].groupby("frame").size()

    # frames where element almost disappears (severe deterioration)
    missing_frames = per_frame_counts[per_frame_counts == 0].index.tolist()

    if missing_frames:
        missing_log.append({
            "class": c,
            "missing_at_frames": missing_frames[:10]  # preview few frames
        })

missing_df = pd.DataFrame(missing_log)

if not missing_df.empty:
    missing_df.to_csv("missing_elements.csv", index=False)
    print("\nSaved: missing_elements.csv")
    print("\nMissing detections found:")
    print(missing_df)
else:
    print("\nNo missing elements detected.")

print("\nAnalysis complete.")
