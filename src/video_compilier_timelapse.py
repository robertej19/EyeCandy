import cv2
import os
import glob
from datetime import datetime

def main():
    # Directory containing the JPEG frames (filenames are Unix timestamps in ms).
    frames_dir = "frames"
    file_pattern = os.path.join(frames_dir, "*.jpg")
    
    # Get a sorted list of all jpg files, sorted by their numeric timestamp (from the filename).
    files = sorted(glob.glob(file_pattern), 
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if not files:
        print("No frames found in the directory.")
        return

    # Get the timestamp (in seconds) of the first frame.
    first_ts = int(os.path.splitext(os.path.basename(files[0]))[0]) / 1000.0
    
    # For a 50x speedup:
    # One second of output (30 frames at 30 FPS) represents 50 seconds of real time.
    # Therefore, each output frame should represent 50/30 seconds (~1.667 sec) of capture time.
    desired_interval = 100.0 / 30.0  # seconds per output frame
    next_required_time = first_ts  # initialize with the first frame's time
    
    # Read the first frame to determine the video dimensions.
    first_frame = cv2.imread(files[0])
    if first_frame is None:
        print("Error reading the first frame.")
        return
    height, width, _ = first_frame.shape

    # Set up VideoWriter to produce an MP4 file at 30 FPS.
    output_file = "output_100x_30fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    print("Assembling video with approximately 50x speedup (output at 30 FPS).")

    # Process and select frames by their timestamp.
    for f in files:
        # Extract timestamp (in seconds) from the filename.
        ts_int = int(os.path.splitext(os.path.basename(f))[0])
        ts = ts_int / 1000.0
        
        if ts >= next_required_time:
            # Read the frame.
            frame = cv2.imread(f)
            if frame is None:
                continue
            # Convert the frame from BGR to RGB if needed (not necessary for saving).
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Prepare the timestamp string:
            dt = datetime.fromtimestamp(ts)
            # Format as "MM-DD-HH-MM-SS.mmm"
            formatted_time = dt.strftime("%m-%d-%H-%M-%S") + "." + f"{dt.microsecond // 1000:03d}"
            
            # Determine text size and position.
            margin = 10  # pixels from edge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(formatted_time, font, font_scale, thickness)
            x = width - margin - text_width
            y = height - margin

            # Draw the timestamp on the frame in red (BGR: (0, 0, 255)).
            cv2.putText(frame, formatted_time, (x, y), font, font_scale, (0, 0, 255), thickness)

            # Write the processed frame to the video.
            video_out.write(frame)
            print(f"Selected frame at {ts:.2f} sec ({formatted_time})")
            
            # Update the next required timestamp.
            next_required_time += desired_interval

    video_out.release()
    print(f"Video saved as {output_file}")

if __name__ == "__main__":
    main()
