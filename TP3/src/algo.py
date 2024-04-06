import cv2
import numpy as np
import os
import csv
import time
from sklearn.neighbors import KDTree

def extract_descriptor(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_descriptor_of_one_channel(frame, channel=0):
    color_channel = frame[:, :, channel]
    hist = cv2.calcHist([color_channel], [channel], None, [8], [0, 256])
    return cv2.normalize(hist, hist).flatten()

def index_all_videos(path_videos):
    global_index = []
    timestamps = []

    for video_file in sorted(os.listdir(path_videos)):
        print(f"Indexing video {video_file}...")
        video_path = os.path.join(path_videos, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = 10

        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                descriptor = extract_descriptor(frame)
                global_index.append(descriptor)
                time_stamp = i / fps
                sec_ms = f"{int(time_stamp)}.{int((time_stamp - int(time_stamp)) * 1000):03d}"
                timestamps.append(f"{video_file}_{sec_ms}")
        cap.release()

    global_kdtree = KDTree(np.array(global_index), leaf_size=10, metric='euclidean')
    return global_kdtree, timestamps

def search_image(path_image, global_kdtree, timestamps, threshold=0.2):
    query_descriptor = extract_descriptor(cv2.imread(path_image))
    distances, indices = global_kdtree.query([query_descriptor], k=1)
    min_distance = distances[0][0]
    best_match = indices[0][0]

    if min_distance <= threshold:
        return timestamps[best_match].split('_')[0], timestamps[best_match].rsplit('_', 1)[1]
    else:
        return "out", None

def search_all_images(path_images, global_kdtree, timestamps, output_file, output_file_time):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'video_pred', 'minutage_pred'])

    with open(output_file_time, 'w', newline='') as csvfile:
        time_writer = csv.writer(csvfile)
        time_writer.writerow(['image', 'search_time'])

    for image_file in sorted(os.listdir(path_images)):
        image_path = os.path.join(path_images, image_file)
        print(f"\nSearching for image {image_file} in all videos...")
        
        start_search_timer = time.time()
        video_pred, minutage_pred = search_image(image_path, global_kdtree, timestamps)
        search_time = time.time() - start_search_timer

        video_pred = video_pred.split('.')[0]
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if video_pred != "out":
                writer.writerow([image_file.split('.')[0], video_pred, minutage_pred])
            else:
                writer.writerow([image_file.split('.')[0], "out", None])

        with open(output_file_time, 'a', newline='') as csvfile:
            time_writer = csv.writer(csvfile)
            time_writer.writerow([image_file, f"{search_time:.4f}"])

    print("Finished searching images.")
    
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
results_dir = os.path.join(script_dir, '../results')


path_videos = os.path.join(data_dir, 'mp4')
path_images = os.path.join(data_dir, 'jpeg')
output_file = os.path.join(results_dir, 'test.csv')
output_file_time = os.path.join(results_dir, 'time.csv')

print("Indexing videos...")
start_index_timer = time.time()
global_kdtree, timestamps = index_all_videos(path_videos)
index_time = time.time() - start_index_timer
print("Searching images...")
search_all_images(path_images, global_kdtree, timestamps, output_file, output_file_time)
print(f"Indexing took {index_time:.2f} seconds.")