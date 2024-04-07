import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform face detection on an image
def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the image with detected faces
    cv2.imshow('Detected Faces', image)
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
        detect_faces(image)
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
            detect_faces(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid option. Please choose 1 or 2.")

# Call the function to choose the option
choose_option()
