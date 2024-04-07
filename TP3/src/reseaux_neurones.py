import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import csv
import numpy as np
import cv2
import time
from einops import rearrange

#### given dont change
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])
####

# dont change 
def extract_features(frame):
    image = Image.fromarray(frame)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    output = rearrange(output, 'b c h w -> (b c h w)')
    return output.numpy().flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidian_distance(a, b):
    return np.linalg.norm(a - b)

def battacharyya_distance(a, b):
    a_normalized = a / np.sum(a)
    b_normalized = b / np.sum(b)
    return -np.log(np.sum(np.sqrt(a_normalized * b_normalized)))

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
        frame_step = 10

        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                descriptor = extract_features(frame)
                global_index.append(descriptor)
                time_stamp = i / fps
                sec_ms = f"{int(time_stamp)}.{int((time_stamp - int(time_stamp)) * 1000):03d}"
                timestamps.append(f"{video_file}_{sec_ms}")
                n_descriptor += 1
        cap.release()

    return global_index, timestamps, n_descriptor

# threshhold to change 
def search_image(path_image, global_index, timestamps, threshold=0.85):
    start_time = time.time()

    query_descriptor = extract_features(cv2.imread(path_image))
    global_index = np.array(global_index)
    distance = np.array([cosine_similarity(query_descriptor, descriptor) for descriptor in global_index]) # threshhold = 0.85 (94.7 %)
    max_distance = np.max(distance)
    # distance = np.array([battacharyya_distance(query_descriptor, descriptor) for descriptor in global_index])
    # min_distance = np.min(distance)
    search_time = time.time() - start_time

    if max_distance >= threshold:
        best_match = np.argmax(distance)
        video_pred = timestamps[best_match].split('_')[0]
        minutage_pred = timestamps[best_match].rsplit('_', 1)[1]
        return video_pred, minutage_pred, search_time
    else:
        return "out", None, search_time

    # if min_distance <= threshold:
    #     best_match = np.argmin(distance)
    #     video_pred = timestamps[best_match].split('_')[0]
    #     minutage_pred = timestamps[best_match].rsplit('_', 1)[1]
    #     return video_pred, minutage_pred, search_time
    # else:
    #     return "out", None, search_time


def search_all_images(path_images, global_index, timestamps, output_file):

    average_search_time = 0
    nb_images = 0

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'video_pred', 'minutage_pred'])

    for image_file in sorted(os.listdir(path_images)):
        image_path = os.path.join(path_images, image_file)
        print(f"\nSearching for image {image_file} in all videos...")
        
        video_pred, minutage_pred, search_time = search_image(image_path, global_index, timestamps)
        nb_images += 1
        average_search_time += search_time

        video_pred = video_pred.split('.')[0]
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if video_pred != "out":
                writer.writerow([image_file.split('.')[0], video_pred, minutage_pred])
            else:
                writer.writerow([image_file.split('.')[0], "out", None])

    average_search_time /= nb_images
    print(f"Average search time: {average_search_time:.4f} seconds/image")
    print("Finished searching images.")

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
results_dir = os.path.join(script_dir, '../results')


path_videos = os.path.join(data_dir, 'mp4')
path_images = os.path.join(data_dir, 'jpeg')
output_file = os.path.join(results_dir, 'test.csv')

print("Indexing videos...")
start_index_timer = time.time()
global_index, timestamps, n_descriptor = index_all_videos(path_videos)
index_time = time.time() - start_index_timer
search_all_images(path_images, global_index, timestamps, output_file)
print(f"Indexing took {index_time:.2f} seconds.")

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
Tc = calculate_matrix_size(n_descriptor, size_descriptor)
print("storage size of videos: ", To)
print("storage size of matrix: ", Tc)
print("Compression ratio: ", 1 - Tc / To)
