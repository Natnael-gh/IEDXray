import json

# Path to your JSON file
input_json = r"C:\Users\Natnael\Desktop\Datasets\2ndExperiment\device_detection\coco\annotations\instances_val2017.json"
output_json = r"C:\Users\Natnael\Desktop\Datasets\2ndExperiment\device_detection\coco\annotations\instances_val22017.json"
# Open and load JSON
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Go through all images and replace 'test2017' with 'val2017'
for img in data.get("images", []):  
    if "file_name" in img and "test2017" in img["file_name"]:
        img["file_name"] = img["file_name"].replace("test2017", "val2017")

# Save updated JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("✅ Finished updating file names. Saved to", output_json)
