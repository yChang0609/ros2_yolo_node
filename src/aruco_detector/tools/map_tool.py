import cv2
import yaml
import numpy as np
import math
import os

# === 修改這裡 ===
map_path = '/home/test/workspace/ros2_yolo_node/src/aruco_detector/map'
yaml_path = yaml_path = os.path.join(map_path, 'map01.yaml')
with open(yaml_path, 'r') as f:
    map_data = yaml.safe_load(f)

image_path = yaml_path = os.path.join(map_path, map_data['image'])
resolution = map_data['resolution']
origin = map_data['origin']  # [origin_x, origin_y, yaw]
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise RuntimeError(f"無法載入地圖圖像: {image_path}")
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.flip(img, 0)
# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_h, img_w  = img.shape
clicked = False
p1 = None
p2 = None
marker_dict = {}

def to_map_coords(px, py):
    map_x = px * resolution + origin[0]
    map_y = py * resolution + origin[1]

    return map_x, map_y

def mouse_event(event, x, y, flags, param):
    global clicked, p1, p2
    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y)
        clicked = False
    elif event == cv2.EVENT_MOUSEMOVE and p1:
        p2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        clicked = True

cv2.namedWindow("Map")
cv2.setMouseCallback("Map", mouse_event)

while True:
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    if p1 and p2:
        cv2.circle(disp, p1, 5, (0, 0, 255), -1)
        cv2.arrowedLine(disp, p1, p2, (0, 255, 0), 2)

    cv2.imshow("Map", disp)
    key = cv2.waitKey(10)

    if key == 27:  # ESC
        print("\nArUco Map Dict:")
        print(yaml.dump(marker_dict, sort_keys=False))
        break
    elif key in range(ord('0'), ord('9') + 1) and clicked and p1 and p2:
        marker_id = chr(key)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = math.atan2(dy, dx)
        map_x, map_y = to_map_coords(p1[0], p1[1])
        marker_dict[int(marker_id)] = {
            'x': round(map_x, 3),
            'y': round(map_y, 3),
            'theta': round(theta, 3)
        }
        print(f"已儲存 marker ID {marker_id}:x={map_x:.3f}, y={map_y:.3f}, theta={theta:.3f} rad")
        p1, p2 = None, None  # 重置
        clicked = False

cv2.destroyAllWindows()