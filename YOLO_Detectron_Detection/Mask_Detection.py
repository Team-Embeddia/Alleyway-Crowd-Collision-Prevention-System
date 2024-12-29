import cv2
import torch
import json
import os
import math
import time
import requests
import numpy as np
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from sklearn.cluster import DBSCAN

def load_yolov8_model(device):
    print("Loading YOLOv8x")
    model = YOLO("yolov8x.pt")
    model.to(device)
    print("YOLOv8x Ready on CPU.")
    return model

def load_detectron2_model(device):
    print("Loading Detectron2 with enhanced configuration")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    predictor.model.to(device)
    print("Detectron2 Ready on CPU.")
    return predictor

def detect_people_yolov8(model, img, conf_threshold=0.3):
    results = model(img)
    persons_locations = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            if conf >= conf_threshold and cls == 0:
                x1, y1, x2, y2 = xyxy
                persons_locations.append((x1.item(), y1.item(), (x2-x1).item(), (y2-y1).item()))
    return persons_locations

def detect_head_or_upper_body(predictor, img):
    outputs = predictor(img)
    instances = outputs["instances"]
    heads_or_upper_bodies = []
    if "pred_masks" in instances.get_fields():
        masks = instances.pred_masks.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        for mask, cls in zip(masks, classes):
            if cls in [0, 1]:
                mask_area = np.sum(mask)
                if mask_area > 100:
                    heads_or_upper_bodies.append(mask)
    return heads_or_upper_bodies

def merge_detections(persons_locations, masks, iou_threshold=0.4):
    merged_locations = persons_locations.copy()
    for mask in masks:
        mask_box = cv2.boundingRect(mask.astype(np.uint8))
        for px, py, pw, ph in persons_locations:
            x1, y1, x2, y2 = mask_box
            if iou((px, py, pw, ph), (x1, y1, x2 - x1, y2 - y1)) > iou_threshold:
                break
        else:
            merged_locations.append(mask_box)
    return merged_locations

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_centroids(merged_locations):
    centroids = []
    for x, y, w, h in merged_locations:
        centroid_x = x + w / 2
        centroid_y = y + h / 2
        centroids.append([centroid_x, centroid_y])
    return np.array(centroids)

def calculate_density_with_dbscan(centroids, eps=15, min_samples=4):
    if len(centroids) == 0:
        return 0, 0, []
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(centroids)

    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    outliers = np.sum(labels == -1)

    cluster_sizes = [np.sum(labels == i) for i in range(num_clusters)]
    return num_clusters, outliers, cluster_sizes

def calculate_status_with_dbscan(num_clusters, cluster_sizes, density_threshold=10):
    if any(size >= density_threshold for size in cluster_sizes):
        return "High"
    elif num_clusters > 1 and max(cluster_sizes) >= density_threshold // 2:
        return "Medium"
    return "Low"

def visualize_and_save(img, merged_locations, status, density, people_count, masks):
    for mask in masks:
        color_mask = np.zeros_like(img)
        color_mask[mask] = (0, 0, 255)
        img = cv2.bitwise_or(img, color_mask)

    for (x, y, w, h) in merged_locations:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    cv2.putText(img, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"People Count: {people_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite("output.png", img)
    print("Saved as 'output.png'")

def visualize_centroids(img, centroids):
    for centroid in centroids:
        cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 255), -1)
    cv2.imwrite("centroids.png", img)
    print("Saved centroids as 'centroids.png'")

def load_image_from_url(image_url, retries=10, delay=5):
    """Fetches an image from a URL and loads it as a numpy array."""
    retry_count = 0
    while retry_count < retries:
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"Successfully loaded image from URL: {image_url}")
                    return img
                else:
                    print(f"Failed to decode image from URL, attempt {retry_count + 1}/{retries}")
            else:
                print(f"HTTP error {response.status_code} from URL, attempt {retry_count + 1}/{retries}")
        except Exception as e:
            print(f"Error fetching image: {e}, attempt {retry_count + 1}/{retries}")
        
        retry_count += 1
        time.sleep(delay)

    print(f"Failed to load image after {retries} attempts, returning a placeholder image.")
    return np.zeros((480, 640, 3), dtype=np.uint8)

def main_loop_with_dbscan(yolov8_model, detectron2_model, image_url):
    while True:
        img = load_image_from_url(image_url)
        if img is None:
            continue

        yolov8_persons_locations = detect_people_yolov8(yolov8_model, img)
        detectron2_masks = detect_head_or_upper_body(detectron2_model, img)

        yolov8_person_count = len(yolov8_persons_locations)
        detectron2_person_count = len(detectron2_masks)

        average_people_count = (yolov8_person_count + detectron2_person_count) / 2
        rounded_people_count = round(average_people_count)

        merged_locations = merge_detections(yolov8_persons_locations, detectron2_masks)

        if len(merged_locations) == 0:
            print("No valid detections found.")
            continue

        centroids = extract_centroids(merged_locations)
        
        if len(centroids) == 0:
            print("No centroids found. Skipping DBSCAN.")
            continue

        visualize_centroids(img, centroids)

        num_clusters, outliers, cluster_sizes = calculate_density_with_dbscan(centroids)
        status = calculate_status_with_dbscan(num_clusters, cluster_sizes)

        save_results(rounded_people_count, status, sum(cluster_sizes))
        visualize_and_save(img, merged_locations, status, len(centroids), rounded_people_count, detectron2_masks)
        print(f"Saved results: {rounded_people_count} people, Status: {status}, Clusters: {num_clusters}")

        time.sleep(5)

def save_results(people_count, status, density):
    results = {
        "people_count": int(people_count),
        "status": status
    }
    with open("shared_data.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved as 'shared_data.json'")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolov8_model = load_yolov8_model(device)
    detectron2_model = load_detectron2_model(device)
    
    image_url = "http://100.123.108.91:5001/get_image"
    main_loop_with_dbscan(yolov8_model, detectron2_model, image_url)