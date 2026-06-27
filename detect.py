from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train-3/weights/best.pt")

# Run detection
results = model(
    "dataset/test/images/P1.jpg",
    conf=0.1,
    imgsz=640
)

# Draw detections
output = results[0].plot()

# Save output
cv2.imwrite("output.jpg", output)

# Resize for display
display = cv2.resize(output, (1200, 800))

# Show image
cv2.imshow("Leather Defect Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection completed")