import cv2
import numpy as np

def process_video_with_mask(video_path, output_path=None, threshold_value=110):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Set up the output video writer if needed
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # Define the hatched regions as rectangles based on your parameters
    rect_width = int(0.47 * frame_width)   
    rect_height = int(0.4 * frame_height)  
    left_rect_x = int(0 * frame_width)   
    right_rect_x = int(0.6 * frame_width)  
    rect_y = int(0.6 * frame_height)      

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create a mask to exclude the rectangular regions
        mask = np.ones_like(gray_frame, dtype=np.uint8) * 255  # Start with a white mask
        cv2.rectangle(mask, (left_rect_x, rect_y), 
                      (left_rect_x + rect_width, rect_y + rect_height), 0, -1)  # Black out the left region
        cv2.rectangle(mask, (right_rect_x, rect_y), 
                      (right_rect_x + rect_width, rect_y + rect_height), 0, -1)  # Black out the right region
        
        # Apply the mask to the grayscale image
        masked_frame = cv2.bitwise_and(gray_frame, mask)
        
        # Apply thresholding to isolate the object of interest
        _, thresholded_frame = cv2.threshold(masked_frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Show the result
        cv2.imshow('Thresholded Frame with Mask', thresholded_frame)

        # Write the frame to the output video if saving
        if output_path is not None:
            out.write(thresholded_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()

# Paths
video_path = '/Users/yanis/Desktop/N=0 (croped).MP4'  # Path to your video file
output_path = '/Users/yanis/Desktop/image_processing/output_with_mask.MP4'  # Path to save the processed video

# Run the function
process_video_with_mask(video_path, output_path)
