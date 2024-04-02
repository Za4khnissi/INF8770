import cv2
import numpy as np
import os
from scipy.spatial import distance
import csv

def index_video(file_path, n_images):
    index = {}
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_step = max(frame_count // n_images, 1)

    for i in range(0, frame_count, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            time_stamp = i / fps
            sec_ms = f"{int(time_stamp)}.{int((time_stamp - int(time_stamp)) * 1000):03d}"
            index[f"{file_path}_{sec_ms}"] = hist
    cap.release()
    return index

def search_image(path_image, index, threshold, max_threshold):
    image = cv2.imread(path_image)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    min_distance = float('inf')
    best_match = None

    for (k, hist_index) in index.items():
        d = distance.euclidean(hist_index, hist)
        if d < min_distance:
            min_distance = d
            best_match = k
        # Early termination if distance is already above the maximum threshold
        if min_distance > max_threshold:
            return "out", None

    if best_match and min_distance <= threshold:
        return best_match.split('_')[0], best_match.rsplit('_', 1)[1]
    else:
        return "out", None

def search_all_images(path_images, path_videos, n_images, threshold, max_threshold, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'video', 'minutage'])

    for image_file in sorted(os.listdir(path_images)):
        found_match = False
        image_path = os.path.join(path_images, image_file)
        print(f"\nRecherche pour l'image {image_file} dans toutes les vidéos...")

        for video_file in sorted(os.listdir(path_videos)):
            video_path = os.path.join(path_videos, video_file)
            print(f"   Vérification dans la vidéo {video_file}...")
            index = index_video(video_path, n_images)
            video_match, timestamp = search_image(image_path, index, threshold, max_threshold)

            if video_match != "out":
                full_video_name = os.path.basename(video_match)
                video_name = os.path.splitext(full_video_name)[0]
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([image_file.split('.')[0], video_name, timestamp])
                found_match = True  
                break 

        if not found_match:
            with open(output_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_file.split('.')[0], "out", None])

path_videos = "C:/Users/zkhni/Desktop/Github Repo/INF8770/TP3/data/mp4"
path_images = "C:/Users/zkhni/Desktop/Github Repo/INF8770/TP3/data/jpeg"
n_images = 7
threshold = 0.5
max_threshold = 0.7
output_file = "C:/Users/zkhni/Desktop/Github Repo/INF8770/TP3/results/test.csv"

search_all_images(path_images, path_videos, n_images, threshold, max_threshold, output_file)