import cv2
import numpy as np

# 門的範圍
lower_door = np.array([80, 123, 180])
upper_door = np.array([97, 139, 184])

# 地板的範圍
lower_floor = np.array([84, 78, 73])
upper_floor = np.array([122, 116, 111])

# 牆的範圍
lower_wall = np.array([170, 152, 99])
upper_wall = np.array([174, 156, 103])

# 讀入影像
img = cv2.imread('detect.png')

# 建立遮罩
mask_door = cv2.inRange(img, lower_door, upper_door)
mask_floor = cv2.inRange(img, lower_floor, upper_floor)
mask_wall = cv2.inRange(img, lower_wall, upper_wall)

# 可視化
result = img.copy()
result[mask_door > 0] = [0, 255, 0]   # 門：綠色
result[mask_wall > 0] = [0, 0, 255]   # 牆：紅色
result[mask_floor > 0] = [255, 0, 0]  # 地板：藍色

cv2.imshow("Color Classified", result)
cv2.waitKey(0)