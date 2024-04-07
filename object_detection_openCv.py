import cv2
import numpy as np

# Load the pre-trained SSD model for object detection
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco.pbtxt')

# Function to perform object detection on an image
def detect_objects(image):
    # Resize the image to a fixed size (300x300) for SSD model
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

    # Set the input for the network
    net.setInput(blob)

    # Run forward pass to perform object detection
    detections = net.forward()

    # Loop over the detections and draw bounding boxes around detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to choose between image from PC or live webcam feed
def choose_option():
    option = input("Choose an option:\n1. Image from PC\n2. Live webcam feed\nEnter option (1/2): ")
    if option == '1':
        # Load image from PC
        img_path = input("Enter the path to the image file: ")
        image = cv2.imread(img_path)
        if image is None:
            print("Error: Unable to read image file.")
            return
        detect_objects(image)
    elif option == '2':
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            detect_objects(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid option. Please choose 1 or 2.")

# Call the function to choose the option
choose_option()
