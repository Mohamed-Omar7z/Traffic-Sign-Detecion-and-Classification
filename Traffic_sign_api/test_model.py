from ultralytics import YOLO

# Load your trained model
model = YOLO("models/best.pt")

# Run prediction on a test image (replace with a real image path on your machine)
results = model.predict(source="test_image.jpg", conf=0.25)

# Print out what it found
for r in results:
    print("Detected boxes:", r.boxes.xyxy)      # bounding box coordinates
    print("Classes:", r.boxes.cls)               # class indices
    print("Confidences:", r.boxes.conf)           # confidence scores

print("Model loaded and ran successfully.")