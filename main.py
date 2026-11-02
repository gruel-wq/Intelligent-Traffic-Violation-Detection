from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Vehicle class IDs
vehicle_classes = [2, 3, 5, 7]

# Detect only vehicles
results = model.predict(
    source="test_images/bus.jpg",
    classes=vehicle_classes,
    conf=0.25,
    save=True
)

# Count detected vehicles
vehicle_count = 0

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        object_name = model.names[class_id]

        vehicle_count += 1
        print("Detected vehicle:", object_name)

print("Total vehicles:", vehicle_count)