import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import glob
import joblib 

# 儲存人工標記資料
samples = []
labels = []

label_colors = {
    0: (0, 255, 0),     # 地板 → 綠
    1: (255, 0, 0),     # 牆 → 藍
    2: (0, 0, 255),     # 柱子 → 紅
    3: (0, 255, 255),   # 門 → 黃
}

label_names = {
    0: "door",
    1: "wall",
    2: "floor",
    3: "pillar"
}

current_label = 0
polygon_points = []
drawing_polygon = False

def mouse_callback(event, x, y, flags, param):
    global current_label, polygon_points, drawing_polygon
    img, display = param

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        drawing_polygon = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右鍵：完成多邊形並灑點
        if len(polygon_points) >= 3:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(polygon_points)], 255)

            num_points = 500  # 控制灑點密度
            ys, xs = np.where(mask == 255)
            coords = list(zip(xs, ys))

            np.random.shuffle(coords)
            coords = coords[:num_points]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            for x_s, y_s in coords:
                hsv_pixel = hsv[y_s, x_s]
                samples.append(hsv_pixel)
                labels.append(current_label)
                cv2.circle(display, (x_s, y_s), 2, label_colors[current_label], -1)

            print(f"已從多邊形加入 {len(coords)} 個樣本點（{label_names[current_label]}）")

        polygon_points = []
        drawing_polygon = False

def label_images(image_paths):
    global current_label

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        display = img.copy()
        cv2.namedWindow("Labeling")
        cv2.setMouseCallback("Labeling", mouse_callback, param=(img, display))

        print("操作說明：")
        print(" - 左鍵點選多邊形")
        print(" - 右鍵完成多邊形並自動取樣")
        print(" - 按 0: 門, 1: 牆, 2: 地板, 3: 柱子 s: 儲存下一張，ESC: 跳過")

        while True:
            if len(polygon_points) >= 1:
                for pt in polygon_points:
                    cv2.circle(display, pt, 3, (255, 255, 255), -1)
                for i in range(1, len(polygon_points)):
                    cv2.line(display, polygon_points[i - 1], polygon_points[i], (255, 255, 255), 1)

            cv2.imshow("Labeling", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('0'):
                current_label = 0
                print("目前標記類別：門")
            elif key == ord('1'):
                current_label = 1
                print("目前標記類別：牆")
            elif key == ord('2'):
                current_label = 2
                print("目前標記類別：地板")
            elif key == ord('3'):
                current_label = 3
                print("目前標記類別：柱子")
            elif key == ord('s'):
                print("儲存並前往下一張")
                break
            elif key == 27:  # ESC
                print("跳過這張")
                break

        cv2.destroyWindow("Labeling")

def train_kmeans(samples, n_clusters=3, save_path="kmeans_model.pkl"):
    if len(samples) == 0:
        print("[Error] 樣本為空，請先標記後再訓練")
        return None
    print(f"\n開始訓練 KMeans樣本數{len(samples)}")
    X = np.array(samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    # joblib.dump(kmeans, save_path)
    np.save("kmeans_centers.npy", kmeans.cluster_centers_)
    print(f"已儲存模型至：{save_path}")
    return kmeans

def predict_and_display(image_path, centers, threshold=350.0):  # 加入 threshold 距離
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    reshaped = hsv.reshape(-1, 3)

    dists = np.linalg.norm(reshaped[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    pred_labels = np.argmin(dists, axis=1)

    pred_labels[min_dists > threshold] = 4

    extended_label_colors = label_colors.copy()
    extended_label_colors[4] = (160, 160, 160)

    colored = np.zeros_like(reshaped)
    for i, label in enumerate(pred_labels):
        colored[i] = extended_label_colors.get(label, (255, 255, 255))

    result = colored.reshape(h, w, 3)

    cv2.imshow("Original", img)
    cv2.imshow("KMeans HSV Classification (with Other)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def load_kmeans_centers(path="kmeans_centers.npy"):
    if not os.path.exists(path):
        print(f"[Error] 中心點檔案不存在：{path}")
        return None
    centers = np.load(path)
    print(f"已載入中心點：{path}")
    return centers

if __name__ == "__main__":
    image_paths = sorted(glob.glob("input_images/*.png"))
    test_image_path = "test.png"
    model_path = "kmeans_centers.npy"

    retrain = False 

    if retrain:
        label_images(image_paths)
        train_kmeans(samples, n_clusters=5, save_path=model_path)
        centers = np.load(model_path)
    else:
        centers = load_kmeans_centers(model_path)

    if centers is not None:
        predict_and_display(test_image_path, centers)