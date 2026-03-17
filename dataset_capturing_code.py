import cv2
import os
import time

# Set DroidCam stream URL
cap = cv2.VideoCapture("http://192.168.31.180:4747/video")

# Directory to save images (NOT home root)
save_dir = "/home/hello1/cam_data/images_100"
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
SAVE_FPS = 3  # Save 3 images per second
delay = 1 / SAVE_FPS

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    filename = os.path.join(save_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")
    frame_count += 1

    cv2.imshow("DroidCam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(delay)

cap.release()
cv2.destroyAllWindows()

