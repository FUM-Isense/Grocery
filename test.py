import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the color stream
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

# Start the pipeline
pipeline.start(config)

# Get the active device and configure auto exposure
device = pipeline.get_active_profile().get_device()
color_sensor = device.query_sensors()[1]  # Assuming color sensor is at index 1
color_sensor.set_option(rs.option.enable_auto_exposure, False)

# Start the OpenCV window
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize a counter for saved frames
frame_count = 0

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert image to numpy array
        np_frame = np.asanyarray(color_frame.get_data())
        color_image = cv2.rotate(np_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Display the image in an OpenCV window
        cv2.imshow('RealSense', color_image)

        # Save the frame when spacebar is pressed
        key = cv2.waitKey(1)
        if key == 32:  # ASCII value for spacebar is 32
            frame_count += 1
            filename = f'saved_frame_{frame_count}.png'
            cv2.imwrite(filename, color_image)
            print(f"Frame saved as '{filename}'")

        # Exit on pressing 'q'
        if key == ord('q'):
            break

finally:
    # Stop the pipeline and close the OpenCV window
    pipeline.stop()
    cv2.destroyAllWindows()
