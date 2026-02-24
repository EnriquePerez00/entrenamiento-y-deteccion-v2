import cv2
import numpy as np

# A basic test of an angle calculated from a rotated bounding box
points = np.array([[10, 10], [50, 20], [40, 60], [0, 50]], dtype=np.float32)
rect = cv2.minAreaRect(points)
box = cv2.boxPoints(rect) 
angle = rect[2]
print(f"Angle: {angle}")

center = (25, 35) # approx center
M = cv2.getRotationMatrix2D(center, angle, 1.0)
print(f"Rotation Matrix:\n{M}")
