# -- color_detect.py --
import cv2
import numpy as np

# 儲存游標位置
cursor_pos = (0, 0)

# 門的範圍
lower_door = np.array([64, 120, 180])
upper_door = np.array([99, 139, 195])

# 地板的範圍
lower_floor = np.array([81, 75, 70])
upper_floor = np.array([128, 123, 111])

# 牆的範圍
lower_wall = np.array([165, 145, 95])
upper_wall = np.array([174, 160, 103])

def mouse_callback(event, x, y, flags, param):
    global cursor_pos
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_pos = (x, y)

def main():
    image_path = "/workspaces/src/door_detector/tools/detect.png"  # TODO: 改成你的影像檔路徑
    img = cv2.imread(image_path)
    if img is None:
        print("[Error] Failed to load image.")
        return

    window_name = "Color Detector"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    # 建立遮罩
    mask_door = cv2.inRange(img, lower_door, upper_door)
    mask_floor = cv2.inRange(img, lower_floor, upper_floor)
    mask_wall = cv2.inRange(img, lower_wall, upper_wall)

    # 可視化
    img[mask_door > 0] = [0, 255, 0]   # 門：綠色
    img[mask_wall > 0] = [0, 0, 255]   # 牆：紅色
    img[mask_floor > 0] = [255, 0, 0]  # 地板：藍色
    while True:
        display = img.copy()

        x, y = cursor_pos
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            b, g, r = img[y, x]
            color_text = f"BGR: ({b}, {g}, {r})"
            # 顯示顏色標籤
            cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(display, color_text, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(10)
        if key == 27:  # 按下 ESC 離開
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
