# import os
# import cv2

# video_path = './data/videos/'
# output_folder = './data/images/'

# # open video
# i = 1
# video_file_name = video_path + 'video_' + str(i) + '.mp4'
# cap = cv2.VideoCapture(video_file_name)
# fps_video = cap.get(cv2.CAP_PROP_FPS) # the original fps of the video

# # you have to calculate the time interval between frames to extract 5 frames per second
# # for example, if the original fps is 30 so we get 1 frame every 6 frames (30 / 5 = 6)
# skip_frames = round(fps_video / 5)

# count = 0
# saved_count = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # get the exactly 5 frames per second
#     if count % skip_frames == 0:
#         image_name = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
#         cv2.imwrite(image_name, frame)
#         saved_count += 1
    
#     count += 1

# cap.release()


import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, target_fps = 5, extension = 'jpg'):
    # kiem tra file video dau vao
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return
    
    # create output folder if not exists
    output_path = Path(output_dir)
    output_path.mkdir(parents = True, exist_ok = True)
    
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: can not open video file {video_path}")
        return
    
    # get fps information of the original video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print(f"Error: can not get the fps of the original video")
        return
    
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Target FPS: {target_fps} frames per second")
    
    # tinh toan khoang cach giua cac frame can lay
    # vi du: neu video 30fps, lay 5fps -> cu moi 6 frame lay 1 frame (30 / 5 = 6)
    hop_interval = max(1, int(round(original_fps / target_fps)))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # chi luu neu frame hien tai nam trong khoang interval da tinh
        if frame_count % hop_interval == 0:
            file_name = f"frame_{saved_count:05d}.{extension}"
            save_path = output_path / file_name
            
            # luu anh
            cv2.imwrite(str(save_path), frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...", end = '\r')
        
        frame_count += 1
    
    cap.release()
    print(f"\nDone! Total frames saved: {saved_count} in folder {output_dir}")

if __name__ == "__main__":
    # thiet lap tham so dong lenh
    parser