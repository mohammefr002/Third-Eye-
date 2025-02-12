# Import necessary libraries
from PIL import Image  # Import the Python Imaging Library (PIL) for image processing
import pytesseract  # Import pytesseract for Optical Character Recognition (OCR)
import cv2  # Import OpenCV for image processing tasks
from Models.audio import say  # Import the say function for audio output
import os  # Import os for operating system functionalities

# Configuration for Tesseract OCR directory
tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'

# Set the image path and preprocessing type
image_path = "Samples/live.png"  # Path to the input image to be OCR'd
preprocess_type = "blur"  # Type of preprocessing to be done ("thresh" or "blur")

# Initialize webcam capture
cam = cv2.VideoCapture(0)

# Counter for capturing frames
ct = 0

# Capture frames until a certain condition is met
while(1):
    _, img = cam.read()  # Read a frame from the webcam
    cv2.imshow('live', img)  # Display the live frame
    ct += 1  # Increment frame counter

    # Break the loop if ESC key is pressed or a certain number of frames are captured
    if(cv2.waitKey(1) & 0xFF == 27) or (ct >= 500):
        cv2.imwrite('Samples/live.png', img)  # Save the captured frame as 'live.png'
        break

# Read the example image and convert it to grayscale
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the image based on the specified preprocessing method
if preprocess_type == "thresh":  # Apply thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif preprocess_type == "blur":  # Apply median blurring
    gray = cv2.medianBlur(gray, 3)

# Write the grayscale image to disk as a temporary file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# Perform OCR using Tesseract on the temporary file
mytext = pytesseract.image_to_string(Image.open(filename), config=tessdata_dir_config)

# Remove the temporary file
os.remove(filename)

# Print the recognized text and output it as audio
print(mytext)
say(mytext)

# Close all OpenCV windows
cv2.destroyAllWindows()
