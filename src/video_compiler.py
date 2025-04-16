import cv2
import os
import glob
from datetime import datetime

def main():
    # Directory containing the saved frames (filenames are Unix timestamps in milliseconds).
    frames_dir = "frames"
    file_pattern = os.path.join(frames_dir, "*.jpg")
    # Get a sorted list of all jpg files, sorting by filename parsed as an integer.
    files = sorted(glob.glob(file_pattern), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    if not files:
        print("No frames found in the directory.")
        return

    # Read the first image to get frame dimensions.
    first_frame = cv2.imread(files[0])
    if first_frame is None:
        print("Error reading the first frame.")
        return
    height, width, _ = first_frame.shape

    # Extract timestamps from filenames to compute the effective FPS.
    timestamps = []
    for f in files:
        ts_int = int(os.path.splitext(os.path.basename(f))[0])
        # Convert milliseconds to seconds.
        timestamps.append(ts_int / 1000.0)
    
    if len(timestamps) >= 2:
        duration = timestamps[-1] - timestamps[0]
        # Compute FPS so that the (n-1) intervals span the duration.
        fps = (len(timestamps) - 1) / duration if duration > 0 else 30
    else:
        fps = 30

    print(f"Computed FPS for the video: {fps:.2f}")

    # Setup VideoWriter: use 'mp4v' codec for mp4 output.
    output_file = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Settings for drawing timestamp text.
    margin = 10  # pixels from the edge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    for f in files:
        # Read the frame (OpenCV returns in BGR).
        frame_rgb = cv2.imread(f)
        if frame_rgb is None:
            continue

        # Convert the frame from BGR to RGB.
        #frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Extract the timestamp from the filename.
        # The filename (without extension) is assumed to be a Unix timestamp in milliseconds.
        ts_int = int(os.path.splitext(os.path.basename(f))[0])
        dt = datetime.fromtimestamp(ts_int / 1000.0)

        # Format the timestamp.
        # Here we use month-day-hour-minute-second.milliseconds.
        # This is a slight extension of "MM-DD-HH-SS.decimals" so that minutes are not lost.
        formatted_time = dt.strftime("%m-%d-%H-%M-%S") + "." + f"{dt.microsecond // 1000:03d}"

        # Determine the size of the text so we can position it in the bottom right.
        (text_width, text_height), baseline = cv2.getTextSize(formatted_time, font, font_scale, thickness)
        x = width - margin - text_width
        y = height - margin

        # Draw the timestamp text on the image.
        # Because our image is now in RGB (R first), red in RGB is (255, 0, 0).
        cv2.putText(frame_rgb, formatted_time, (x, y), font, font_scale, (255, 0, 0), thickness)

        # Convert the frame back to BGR since VideoWriter (and most codecs) expect BGR.
        frame_final = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the video file.
        video_out.write(frame_final)

    video_out.release()
    print(f"Video saved as {output_file}.")

if __name__ == "__main__":
    main()
