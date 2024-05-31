import os
import cv2 as cv

def getFaceBox(net, frame, conf_threshold=0.75):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Load network
faceNet = cv.dnn.readNet(faceModel, faceProto)
faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

# Define the input folder
input_folder = "input"

# Ensure the input folder exists
if not os.path.exists(input_folder):
    print(f"Error: Input folder '{input_folder}' does not exist.")
    exit()

# Define the output folder
output_folder = "crop_input"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(input_folder, filename)
        print(f"Processing {filepath}...")

        # Read image
        frame = cv.imread(filepath)
        if frame is None:
            print(f"Error: Unable to read {filepath}. Skipping...")
            continue

        bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected in", filepath)
            continue

        # Create a directory to store cropped faces if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Crop and save faces
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            # Crop the face

            face = frame[y1:y2, x1:x2]
            # Resize the face to 200x200
            face_resized = cv.resize(face, (200, 200))
    
            output_filename = os.path.join(output_folder, f"{filename.split('.')[0]}_{i}.jpg")
            cv.imwrite(output_filename, face_resized)
            print(f"  Face {i+1} saved as {output_filename}")

print("All faces cropped and saved successfully.")
