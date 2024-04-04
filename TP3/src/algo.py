import cv2
import numpy as np
import os
from scipy.spatial import distance
import csv
import time
from sklearn.neighbors import KDTree

def index_video(file_path, n_images):
    index = []
    timestamps = []
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_step = frame_count // n_images

    for i in range(0, frame_count, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            index.append(hist)
            time_stamp = i / fps
            sec_ms = f"{int(time_stamp)}.{int((time_stamp - int(time_stamp)) * 1000):03d}"
            timestamps.append(f"{file_path}_{sec_ms}")
    cap.release()
    
    kdtree = KDTree(np.array(index), leaf_size=40, metric='euclidean')
    return kdtree, timestamps

def search_image(path_image, kdtree, timestamps, threshold):
    image = cv2.imread(path_image)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Query the KDTree for the nearest neighbor
    dist, ind = kdtree.query([hist], k=1)
    nearest_dist = dist[0][0]
    nearest_index = ind[0][0]

    if nearest_dist <= threshold:
        return timestamps[nearest_index].split('_')[0], timestamps[nearest_index].rsplit('_', 1)[1]
    else:
        return "out", None

def search_all_images(path_images, path_videos, n_images, threshold, output_file, output_file_time):
    total_index_time = 0
    total_search_time = 0

    with open(output_file_time, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'index_time', 'search_time'])

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'video_pred', 'minutage_pred'])

    for image_file in sorted(os.listdir(path_images)):
        found_match = False
        image_path = os.path.join(path_images, image_file)
        print(f"\nRecherche pour l'image {image_file} dans toutes les vidéos...")

        for video_file in sorted(os.listdir(path_videos)):
            video_path = os.path.join(path_videos, video_file)
            print(f"   Vérification dans la vidéo {video_file}...")
            start_index_timer = time.time()

            kdtree, timestamps = index_video(video_path, n_images)
            index_time = time.time() - start_index_timer
            total_index_time += index_time

            start_search_timer = time.time()

            video_match, timestamp = search_image(image_path, kdtree, timestamps, threshold)
            search_time = time.time() - start_search_timer
            total_search_time += search_time

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

        with open(output_file_time, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_file, f"{index_time:.4f}", f"{search_time:.4f}"])

def calculate_storage_size(path_videos):
    total_size = 0
    for video_file in os.listdir(path_videos):
        video_path = os.path.join(path_videos, video_file)
        total_size += os.path.getsize(video_path)
    return total_size

def calculate_matrix_size(n_videos, n_images):
    bytes_per_float = 4
    n_bins = 8 * 8 * 8
    return n_images * n_bins * bytes_per_float * n_videos

def calculate_frame_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_channels = 3
    bit_depth = 1
    
    cap.release()
    return width * height * n_channels * bit_depth

def calculate_total_frame_size(path_videos):
    total_frame_size = 0
    for video_file in os.listdir(path_videos):
        video_path = os.path.join(path_videos, video_file)
        frame_count = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = calculate_frame_size(video_path)
        total_frame_size += frame_size * frame_count
    return total_frame_size

def estimate_jpeg_size(video_path, sample_size=10, jpeg_quality=90):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, sample_size).astype(int)

    total_size = 0
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            _, jpeg_data = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            total_size += len(jpeg_data)

    cap.release()
    average_size = total_size / sample_size
    return average_size, average_size * frame_count

n_images = 7 # empirically determined
threshold = 0.5
max_threshold = 0.7

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
results_dir = os.path.join(script_dir, '../results')


path_videos = os.path.join(data_dir, 'mp4')
path_images = os.path.join(data_dir, 'jpeg')
output_file = os.path.join(results_dir, 'test.csv')
output_file_time = os.path.join(results_dir, 'time.csv')


average_size_per_frame, estimated_total_size = estimate_jpeg_size(path_videos)

total_estimated_size = 0
for video_file in os.listdir(path_videos):
    video_path = os.path.join(path_videos, video_file)
    _, estimated_total_size = estimate_jpeg_size(video_path)
    total_estimated_size += estimated_total_size



n_videos = len(os.listdir(path_videos))
To = calculate_storage_size(path_videos)
Tc = calculate_matrix_size(n_videos, n_images)

total_frame_size = calculate_total_frame_size(path_videos)

compression_rate = (1 - Tc / To)
print(f"Storage size: {To:.2f} bytes")
print(f"Compression rate: {compression_rate:.2f}")
print(f"Total frame size: {total_frame_size:.2f} bytes")
print(f"Estimated total size for all frames of all videos in JPEG: {total_estimated_size} bytes")

# search_all_images(path_images, path_videos, n_images, threshold, output_file, output_file_time)