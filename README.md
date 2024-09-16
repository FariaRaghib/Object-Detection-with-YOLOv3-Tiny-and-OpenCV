# Object Detection with YOLOv3-Tiny and OpenCV
This project demonstrates real-time object detection using YOLOv3-tiny with OpenCV in Python. It uses a webcam to capture video and detect objects using a pre-trained YOLOv3-tiny model. The system is capable of identifying multiple objects in real-time and displaying bounding boxes and class labels.

## Features
1. Real-time Object Detection: Detect multiple objects in real-time using a live camera feed.
2. YOLOv3-Tiny: Utilizes the lighter version of the YOLOv3 model for fast object detection.
3. Class Identification: Displays the class of each detected object (e.g., person, car) with confidence percentages.
4. Non-Max Suppression: Filters out overlapping bounding boxes for more accurate detections.
## Files in the Repository
1. main.py: The main Python script that captures video, processes frames, and detects objects.
2. yolov3_tiny.cfg: The configuration file for the YOLOv3-tiny model.
3. yolov3-tiny.weights: The pre-trained weights file for YOLOv3-tiny.
4. coco.names: The file containing the names of the 80 object classes that YOLO can detect.
## Installation
### Prerequisites
-Python 3.x \
-OpenCV (Python package) \
-NumPy (Python package) 

## Steps to Run
Clone the repository:
### 'git clone https://github.com/FariaRaghib/Object-Detection-with-YOLOv3-Tiny-and-OpenCV'
### 'cd Object-Detection-with-YOLOv3-Tiny-and-OpenCV'

Install the required dependencies:
### 'pip install opencv-python numpy'

Download the pre-trained YOLOv3-tiny weights file (yolov3-tiny.weights)

Run the project:
### 'python Yolo.py'
The system will open a window displaying the camera feed with detected objects.

## How It Works
--> Video Capture: The script opens your webcam (cv2.VideoCapture(0)).

--> Object Detection: Each frame is processed through the YOLO model. The objects are detected and classified based on the COCO dataset.

--> Visualization: Bounding boxes are drawn around detected objects along with their class names and confidence scores.

--> Real-time Feedback: The detection happens in real-time, processing each frame individually.

## Future Improvements
- Multiple Camera Support: Extend the project to support multiple cameras.
- Custom Training: Train YOLO on custom datasets for specific object detection.
- Optimize for Speed: Further optimization to make detection faster on low-end devices.
