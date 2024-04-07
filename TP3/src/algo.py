import cv2
import numpy as np
import os
import csv
import time
from sklearn.neighbors import KDTree

def extract_descriptor(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_1d_descriptors(frame):
    descriptors = []
    
    for channel in range(3):
        hist = cv2.calcHist([frame], [channel], None, [8], [0, 256])
        normalized_hist = cv2.normalize(hist, hist).flatten()
        descriptors.extend(normalized_hist)
    
    return np.array(descriptors)

def index_all_videos(path_videos):
    global_index = []
    timestamps = []
    n_descriptor = 0

    for video_file in sorted(os.listdir(path_videos)):
        print(f"Indexing video {video_file}...")
        video_path = os.path.join(path_videos, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = max(1, int(frame_count / 30))

        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                descriptor = extract_descriptor(frame)
                global_index.append(descriptor)
                time_stamp = i / fps
                sec_ms = f"{int(time_stamp)}.{int((time_stamp - int(time_stamp)) * 1000):03d}"
                timestamps.append(f"{video_file}_{sec_ms}")
                n_descriptor += 1
        cap.release()
    global_kdtree = KDTree(np.array(global_index), leaf_size=10, metric='euclidean')

    return global_kdtree, timestamps, n_descriptor

def search_image(path_image, global_kdtree, timestamps, threshold=0.45):
    start_time = time.time()
    query_descriptor = extract_descriptor(cv2.imread(path_image))
    distances, indices = global_kdtree.query([query_descriptor], k=1)
    min_distance = distances[0][0]
    best_match = indices[0][0]
    search_time = time.time() - start_time

    if min_distance <= threshold:
        return timestamps[best_match].split('_')[0], timestamps[best_match].rsplit('_', 1)[1], search_time
    else:
        return "out", None, search_time

def search_all_images(path_images, global_kdtree, timestamps, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'video_pred', 'minutage_pred'])

    number_of_images_searched = 0
    total_search_time = 0

    for image_file in sorted(os.listdir(path_images)):
        image_path = os.path.join(path_images, image_file)
        print(f"\nSearching for image {image_file} in all videos...")
        
        video_pred, minutage_pred, search_time = search_image(image_path, global_kdtree, timestamps)
        total_search_time += search_time
        number_of_images_searched += 1

        video_pred = video_pred.split('.')[0]
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if video_pred != "out":
                writer.writerow([image_file.split('.')[0], video_pred, minutage_pred])
            else:
                writer.writerow([image_file.split('.')[0], "out", None])

    average_search_time = total_search_time / number_of_images_searched * 1000
    print("Finished searching images.")
    print(f"Average search time: {average_search_time:.4f} ms/image")

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
results_dir = os.path.join(script_dir, '../results')


path_videos = os.path.join(data_dir, 'mp4')
path_images = os.path.join(data_dir, 'jpeg')
output_file = os.path.join(results_dir, 'test.csv')

print("Indexing videos...")
start_time = time.time()
global_kdtree, timestamps, n_descriptors = index_all_videos(path_videos)
indexing_time = time.time() - start_time
print("Searching images...")
search_all_images(path_images, global_kdtree, timestamps, output_file)
print(f"Indexing time: {indexing_time:.4f} seconds")


# Compression ratio
def calculate_storage_size(path_videos):
    total_size = 0
    for video_file in os.listdir(path_videos):
        video_path = os.path.join(path_videos, video_file)
        total_size += os.path.getsize(video_path)
    return total_size

def calculate_matrix_size(n_descriptor, size_descriptor):
    return n_descriptor * size_descriptor * 4

size_descriptor = 8 * 8 * 8
To = calculate_storage_size(path_videos)
Tc = calculate_matrix_size(n_descriptors, size_descriptor)
print("storage size of videos: ", To)
print("storage size of matrix: ", Tc)
print("Compression ratio: ", 1 - Tc / To)



