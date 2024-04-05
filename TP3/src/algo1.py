import cv2
import numpy as np
import os
import csv
import time

def extract_descriptor(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def index_all_videos(path_videos):
    global_index = []
    metadata = []

    for video_file in sorted(os.listdir(path_videos)):
        print(f"Indexing video {video_file}...")
        video_path = os.path.join(path_videos, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frame count: {frame_count}, FPS: {fps}")
        frame_step = 3

        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                descriptor = extract_descriptor(frame)
                global_index.append(descriptor)
                time_stamp = i / fps
                sec_ms = f"{int(time_stamp)}.{int((time_stamp - int(time_stamp)) * 1000):03d}"
                metadata.append(f"{video_file}_{sec_ms}")

    cap.release()
    return global_index, metadata

def search_image(path_image, global_index, metadata, threshold=0.1):
    query_descriptor = extract_descriptor(cv2.imread(path_image))
    min_distance = np.inf
    best_match = None

    # Perform linear scan to find the descriptor with the minimum Euclidean distance
    for i, descriptor in enumerate(global_index):
        distance = np.linalg.norm(descriptor - query_descriptor)
        if distance < min_distance:
            min_distance = distance
            best_match = i

    if min_distance <= threshold:
        match_metadata = metadata[best_match]
        video_name, timestamp = match_metadata.split('_')[0], match_metadata.rsplit('_', 1)[1]
        return video_name, timestamp
    else:
        return "out", None

def search_all_images(path_images, global_index, metadata, output_file, output_file_time):
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
        video_pred, minutage_pred = search_image(image_path, global_index, metadata, threshold=0.1)
        search_time = time.time() - start_search_timer

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
global_index, metadata = index_all_videos(path_videos)
index_time = time.time() - start_index_timer
print("Searching images...")
search_all_images(path_images, global_index, metadata, output_file, output_file_time)
print(f"Indexing took {index_time:.2f} seconds.")