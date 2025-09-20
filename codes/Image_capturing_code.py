# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:45:52 2023

@author: SushmaReenaPaulRaj
"""

# importing cv2 to access OpenCV to solve the computer vision problems
import cv2
# importing os to set directory
import os
# importing time for setting timer
import time

# creating directory for gesture dataset
root_dir = "hand_gesture_dataset"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

train_dir = os.path.join(root_dir, "train")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

test_dir = os.path.join(root_dir, "test")
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
    
# creating sub-directory using gesture name
gestures = ["pause", "play", "volume_up", "volume_down", "previous","next"]
for gesture in gestures:
    train_gesture_dir = os.path.join(train_dir, gesture)
    if not os.path.exists(train_gesture_dir):
        os.mkdir(train_gesture_dir)

    test_gesture_dir = os.path.join(test_dir, gesture)
    if not os.path.exists(test_gesture_dir):
        os.mkdir(test_gesture_dir)

#using OpenCV to capture the image using webcam and save the image
cap = cv2.VideoCapture(0)

# take input for gesture name
gesture = input("Enter gesture name: ")

# take input for train or test directory
is_train = input("Enter 't' for train directory and 'ts' for test directory: ").lower() == 't'
gesture_dir = os.path.join(train_dir if is_train else test_dir, gesture)
if not os.path.exists(gesture_dir):
    os.mkdir(gesture_dir)

# take input for number of photos to capture
num_photos = int(input("enter number of image:"))   

count = 0
start_time = None

# loop until desired number of photos have been captured
while count < num_photos:
    ret, frame = cap.read() # read a frame from camera
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    
    roi = frame[100:300, 100:300] # extract the region of interest
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2) # draw a rectangle around the region of interest
    
    # if timer is not started, display a message to start capturing images
    if start_time is None:
        cv2.putText(frame, f"Press 's' to start capturing images", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # if timer is started, display number of images captured and time left for next capture
    else:
        seconds_left = 2- int(time.time() - start_time)
        if seconds_left < 0:
            seconds_left = 0
        cv2.putText(frame, f"Captured {count} images, {seconds_left}s left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show live feed and region of interest
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Region of Interest", roi)
   
    key = cv2.waitKey(1)
    if key == ord('q'): # quit the program on pressing 'q'
        break
    elif key == ord('s'): # start the timer on pressing 's'
        start_time = time.time()
   
    if start_time is not None and time.time() - start_time >= 2: # capture an image if timer has elapsed for 2 seconds
        filename = os.path.join(gesture_dir, f"{gesture}_{count}.jpg")
        cv2.imwrite(filename, roi)
        count += 1
        start_time = time.time()

# to free camera and allows it to be used by other processes
cap.release()
# to destroy all the windows created by the program to avoid leaving any windows open

cv2.destroyAllWindows()
