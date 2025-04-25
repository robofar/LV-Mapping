import cv2
import os
import sys

# Path to images directory
image_folder = "eval_images_static_sampling"

video_paths = ["front_normal_2.mp4", "left_normal_2.mp4", "right_normal_2.mp4", "rear_normal_2.mp4"] # namesof videos
ends_with = ["front_normal_2.png", "left_normal_2.png", "right_normal_2.png", "rear_normal_2.png"] # names of photos

for i in range(0, len(video_paths)):
    # Get list of images that match the pattern explicitly
    image_files = [
        f for f in os.listdir(image_folder) if f.endswith(ends_with[i])
    ]


    # Sort numerically based on the leading number
    image_files.sort(key=lambda x: int(x.split("_")[0]))  # Extract and sort by numeric prefix


    # Ensure images exist
    if not image_files:
        raise ValueError(f"No valid images found in {image_folder}")



    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, c = first_image.shape

    



    # Define video writer for MP4
    fps = 1  # Adjust as needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4
    video_writer = cv2.VideoWriter(video_paths[i], fourcc, fps, (width, height))

    # Process and write images
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        video_writer.write(img)

    # Release video writer
    video_writer.release()
    print(f"Video saved as {video_paths[i]}")