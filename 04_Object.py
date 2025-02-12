# import the necessary packages
from imutils.video import VideoStream  # Importing VideoStream for accessing the camera feed
import numpy as np  # Importing numpy for numerical computations
import argparse  # Importing argparse for command-line argument parsing
import imutils  # Importing imutils for image processing utilities
import time  # Importing time for time-related functions
import cv2  # Importing OpenCV library for computer vision tasks
import os  # Importing os for operating system functionalities
from Models.audio import say  # Importing a custom module 'say' for speech synthesis

# Saying "Object detection" using the 'say' module
say("Object detection")

# Path to the prototxt file defining the network architecture
prototxt = "Models/deploy.prototxt.txt"
# Path to the pre-trained caffemodel file containing learned parameters
model = "Models/deploy.caffemodel"
# Confidence threshold for filtering out weak detections
confidence_threshold = 0.75

# List of class labels MobileNet SSD was trained to detect
CLASSES = ["None", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "plant", "sheep",
        "sofa", "train", "monitor"]
# Generating a set of bounding box colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Start the video stream, allowing the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Allow a brief delay for the camera sensor to warm up
time.sleep(0.1)

# Initialize variables for storing text displayed on images and detected objects
img_text = ''
found = set()

# Loop over the frames from the video stream
while True:
        # Grab the frame from the video stream and flip it horizontally
        frame = vs.read()
        frame=cv2.flip(frame,1)
      
        # Grab the frame dimensions and convert it to a blob for processing
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

        # Set the blob as input to the network and obtain detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
                # Extract the confidence associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring confidence is above the threshold
                if confidence > confidence_threshold:
                        # Extract the class index and bounding box coordinates
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Draw the bounding box and label on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        
                        # Print the detected object label
                        print(format(CLASSES[idx]))

                        # Perform speech synthesis to announce the detected object
                        if(format(CLASSES[idx]) != str(img_text)):
                                img_text= format(CLASSES[idx])
                                if str(img_text) not in found :
                                        say(str(img_text))
                                        found.add(str(img_text))
                        else:
                                found.clear()

        # Display the frame with object detection annotations
        cv2.imshow("Image Classifier", frame)

        # Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
                break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
