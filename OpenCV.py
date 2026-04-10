#OpenCV

import os
import cv2

video_folder = "videos"
output_folder = "dataset"  

os.makedirs(output_folder, exist_ok=True)

frame_interval = 120  
count = 0

for video_name in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_name)
    
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:
            frame_resized = cv2.resize(frame, (1920, 1080))
            file_name = os.path.join(output_folder, f"img_{count}.jpg")
            cv2.imwrite(file_name, frame)
            count += 1
        
        frame_id += 1

    cap.release()

print("Frames extraídos com sucesso")