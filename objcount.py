import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Load the image
image = cv2.imread("cars.jpeg")

# Perform object detection
box, label, count = cv.detect_common_objects(
    image,
    confidence=0.5,
    model='yolov4',
    enable_gpu=False
)

# Draw bounding boxes
output = draw_bbox(image, box, label, count)

# Display the output
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide the axes for better visualization
plt.show()

# Print the count of cars
print("Number of cars in this image are " + str(label.count('car')))
