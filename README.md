# Object Detection using OpenCV

This Python script performs object detection using the Single Shot MultiBox Detector (SSD) model in OpenCV. It allows users to choose between selecting an image from their PC or using a live webcam feed for object detection.

## Prerequisites

- Python 3.x
- OpenCV library (`pip install opencv-python`)

## Usage

1. Clone or download this repository to your local machine.

2. Navigate to the directory containing the script (`object_detection.py`).

3. Ensure that you have the pre-trained SSD model files (`frozen_inference_graph.pb` and `ssd_mobilenet_v2_coco.pbtxt`) in the same directory as the script.

4. Run the script by executing the following command in your terminal or command prompt:

5. Follow the on-screen prompts to choose between selecting an image from your PC or using a live webcam feed for object detection.

6. If you choose to select an image from your PC, enter the path to the image file when prompted.

7. If you choose to use a live webcam feed, press 'q' to exit the feed.

## Options

- Option 1: Image from PC
- Option 2: Live webcam feed

## Additional Information

- The script uses the SSD model for object detection. You can replace the pre-trained model files with other compatible models if needed.
- Adjust the confidence threshold (`confidence > 0.5`) in the `detect_objects` function to change the sensitivity of object detection.

