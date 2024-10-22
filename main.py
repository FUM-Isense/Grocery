import pyttsx3
from pynput import keyboard
from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Load the DenseNet model (pre-trained on ImageNet)
class DenseNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(DenseNetFeatureExtractor, self).__init__()
        densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        self.features = densenet.features
    
    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

# Preprocessing transforms for the DenseNet model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image region
def extract_features(model, image, device):
    model.eval()
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        features = model(image)
    return features.cpu().numpy()

# Function to find the best matching box
def find_best_matching_box(sample_image, shelf_image, bounding_boxes, model, device):
    sample_features = extract_features(model, sample_image, device)

    similarities = []
    box_coords = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box[:4]
        cropped_box = shelf_image.crop((x1, y1, x2, y2))
        box_features = extract_features(model, cropped_box, device)
        similarity = cosine_similarity(sample_features, box_features)[0][0]
        similarities.append(similarity)
        box_coords.append(box)

    # print(f'################similarity values{similarities}')
    best_idx = np.argmax(similarities)
    return box_coords[best_idx]

def find_sample(model, frame, counter=1, highlight_color=(0, 255, 0)):
    results = model.track(source=frame, persist=True, tracker="botsort.yaml")
    annotated_frame = np.array(frame.copy())
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            boxes.append((x1, y1, x2, y2, box_area))

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)[:counter]
    for (x1, y1, x2, y2, _) in boxes:
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), highlight_color, 2)
    return boxes, annotated_frame, ((counter != 1) and (len(boxes) == 20))


            
# Function to find the most frequent row and column using your custom logic
def find_best_row_col(best_boxes_list, shelf_boxes_list):   
    pose_ = []
    for index in range(len(best_boxes_list)):
        row, col = 0, 0
        shelf_ = shelf_boxes_list[index]
        box_ = best_boxes_list[index]

        x_sorted_shelf = sorted(shelf_, key=lambda x: x[0], reverse=False)[:]
        y_sorted_shelf = sorted(shelf_, key=lambda x: x[1], reverse=False)[:]

        for j, row_box in enumerate(y_sorted_shelf):
            if box_[1] == row_box[1]:
                row = j // 5
        
        for i, col_box in enumerate(x_sorted_shelf):
            if box_[0] == col_box[0]:
                col = i // 4
    
        pose_.append((row + 1, col + 1))
    print(pose_)
    counter = Counter(pose_)
    # Get the most common element
    most_frequent_pose_, _ = counter.most_common(1)[0]
    return most_frequent_pose_

# Define global variables to control the state of capturing
capture_flag = False
capture_step = 0

def on_press(key):
    global capture_flag, capture_step
    try:
        if key.char == 'b':  # Detect when 'b' is pressed
            capture_flag = True
            capture_step += 1
    except AttributeError:
        pass

# Function to capture frames when 'b' key is pressed
def capture_frames(num_frames, pipeline):
    count = 0
    frames_list = []

    try:
        while count < num_frames:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("No frame received")
                continue

            np_frame = np.asanyarray(color_frame.get_data())
            frame = cv2.rotate(np_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frames_list.append(frame)
            count += 1

    finally:
        return frames_list  # Return the captured frames

if __name__ == "__main__":
    print("-----Main-----")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    model = YOLO("best.pt")

    # Initialize DenseNet feature extractor
    densenet_model = DenseNetFeatureExtractor()

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    densenet_model = densenet_model.to(device)

    engine.say("START")
    engine.runAndWait()
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    pipeline.start(config)
    rs_device = pipeline.get_active_profile().get_device()
    color_sensor = rs_device.query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, True)

    sample_frame = None
    shelf_frames = []

    try:
        while True:
            if capture_flag:
                if capture_step == 1:
                    # Capture a single sample frame
                    print("Capturing sample frame...")
                    engine.say("Capturing frame")
                    engine.runAndWait()
                    sample_frame = capture_frames(1, pipeline)[0]
                    # cv2.imshow("Sample Frame", sample_frame)
                    capture_flag = False  # Reset the flag after capturing
                    engine.say("Ready")
                    engine.runAndWait()

                elif capture_step == 2:
                    # Capture five shelf frames
                    print("Capturing shelf frames...")
                    engine.say("Capturing shelf")
                    engine.runAndWait()
                    shelf_frames = capture_frames(5, pipeline)
                    # for idx, frame in enumerate(shelf_frames):
                    #     cv2.imshow(f"Shelf Frame {idx+1}", frame)
                    capture_flag = False  # Reset the flag after capturing


                    if sample_frame is not None and shelf_frames:
                        sample_pil = Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))

                        # Detect the sample box
                        sample_boxes, annotated_sample, _ = find_sample(model, sample_frame)
                        x1, y1, x2, y2 = sample_boxes[0][:4]
                        sample_roi = sample_pil.crop((x1, y1, x2, y2))

                        best_boxes_list = []
                        shelf_boxes_list = []

                        for img in shelf_frames:
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            shelf_boxes, anotated_shelf, sw = find_sample(model, img, 20)
                            if sw:
                                best_box = find_best_matching_box(sample_roi, img_pil, shelf_boxes, densenet_model, device)
                                best_boxes_list.append(best_box)
                                shelf_boxes_list.append(shelf_boxes)
                        if sw:
                            break
                        else:
                            engine.say("Do it again")
                            engine.runAndWait()
                            capture_step = 0

                            

            # # Quit on 'q' key press
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        listener.stop()


        # Find the best match in terms of row and column
        row, col = find_best_row_col(best_boxes_list, shelf_boxes_list)
        print(f"Most frequent match located at row {row}, column {col}")

        # Display results on the last shelf frame
        matched_shelf_image = cv2.cvtColor(np.array(anotated_shelf), cv2.COLOR_RGB2BGR)
        best_box = best_boxes_list[-1]
        x1, y1, x2, y2 = map(int, best_box[:4])

        # cv2.rectangle(matched_shelf_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.putText(matched_shelf_image, f'Row: {row}, Col: {col}', (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # cv2.imshow("Matched Shelf Image", matched_shelf_image)
        # cv2.waitKey(0)

    # Provide voice feedback
    feedback_message = f"located at row {row}, column {col}."
    engine.say(feedback_message)
    engine.runAndWait()
