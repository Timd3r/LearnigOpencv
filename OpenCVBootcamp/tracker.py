# Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

classFile  = "coco_class_labels.txt"
with open(classFile) as fp:
    labels = fp.read().split("\n")

print(labels)

modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# For each file in the directory
def detect_objects(net, im, dim = 300):

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)

    # Pass blob to the network
    net.setInput(blob)

    # Peform Prediction
    objects = net.forward()
    return objects

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2
THICKNESS = 2

def display_text(im, text, x, y):
    # Get text size
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv2.rectangle(
        im,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Display text inside the rectangle
    cv2.putText(
        im,
        text,
        (x, y - 5),
        FONTFACE,
        FONT_SCALE,
        (0, 255, 255),
        THICKNESS,
        cv2.LINE_AA,
    )

def display_objects(im, objects, threshold=0.25, target_labels=None):
    rows = im.shape[0]
    cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        label = labels[classId]
        
        # Skip labels that aren't in the target_labels list!
        if target_labels is not None and label not in target_labels:
            continue

        # Recover original coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        if score > threshold:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # REMOVED: Matplotlib code that prints frames to the notebook
    return im

# Open the input video
video = cv2.VideoCapture("tracker_test.mp4")

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

# Try different codecs in order of preference
codecs_to_try = [
    ('mp4v', '.mp4'),  # MPEG-4 (most compatible)
    ('XVID', '.avi'),  # XVID
    ('MJPG', '.avi'),  # Motion JPEG
]

working_codec = None
working_ext = None

print("Testing codecs...")
for fourcc_str, ext in codecs_to_try:
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        test_file = f"test_codec{ext}"
        video_out_test = cv2.VideoWriter(test_file, fourcc, fps, (width, height))
        if video_out_test.isOpened():
            video_out_test.release()
            if os.path.exists(test_file):
                os.remove(test_file)
            working_codec = fourcc_str
            working_ext = ext
            print(f"✓ Found working codec: {fourcc_str}")
            break
        else:
            video_out_test.release()
            if os.path.exists(test_file):
                os.remove(test_file)
    except Exception as e:
        print(f"✗ Codec {fourcc_str} failed: {e}")
        continue

if working_codec is None:
    raise Exception("No compatible video codec found!")

# Now create the actual VideoWriter with the working codec
video_output_file_name = f"video-TRACKED{working_ext}"
fourcc = cv2.VideoWriter_fourcc(*working_codec)
video_out = cv2.VideoWriter(video_output_file_name, fourcc, fps, (width, height))

if not video_out.isOpened():
    raise Exception(f"Failed to open VideoWriter with codec {working_codec}")

print("Processing video... Please wait.")


while True:
    ok, frame = video.read()
    if not ok:
        break
    
    objects = detect_objects(net, frame)
    
    # You can change the certainty/confidence threshold here
    # Use target_labels to specify a list of strings of labels you want to track
    display_objects(frame, objects, threshold=0.5, target_labels=["person", "bottle"])
    
    video_out.write(frame)
    

video.release()
video_out.release()

print(f"Done! Video saved as {video_output_file_name}")
print(f"File size: {os.path.getsize(video_output_file_name)} bytes")

import subprocess
import os
from IPython.display import Video

input_file = "video-TRACKED.mp4"
output_file = "video-TRACKED-h264.mp4"

print("Converting video using libopenh264 (browser-compatible)...")
subprocess.run([
    'ffmpeg', '-i', input_file,
    '-c:v', 'libopenh264',   # Changed from libx264 to libopenh264
    '-preset', 'medium',
    '-crf', '23',            # Quality setting (18-28 is good)
    '-c:a', 'aac',
    '-y',                     # Overwrite existing file
    output_file
], check=True)

print(f"Conversion complete! File size: {os.path.getsize(output_file)} bytes")

# Open the video file
video = cv2.VideoCapture("video-TRACKED-h264.mp4")

print("Playing back the processed video...")
cv2.namedWindow("Video Playback", cv2.WINDOW_NORMAL)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
        
    cv2.imshow("Video Playback", frame)
    
    # Wait for ~33ms between frames (approx 30 FPS). Press 'q' to quit.
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
