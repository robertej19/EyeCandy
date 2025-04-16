import requests
import os
import time

# URL of your MJPEG stream
STREAM_URL = "http://192.168.1.190:8485/video"

# Directory where frames will be saved
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

def save_frame(jpg_bytes, timestamp):
    """Save the JPEG frame to disk using the provided timestamp as the filename."""
    filename = os.path.join(FRAME_DIR, f"{timestamp}.jpg")
    with open(filename, "wb") as f:
        f.write(jpg_bytes)

def capture_frames(max_frames=100, batch_size=20):
    # Open the MJPEG stream
    r = requests.get(STREAM_URL, stream=True)
    bytes_data = b""
    frame_count = 0

    # Initialize batch timer for framerate calculation
    last_batch_time = time.time()
    
    for chunk in r.iter_content(chunk_size=1024):
        bytes_data += chunk

        # Look for the start and end JPEG markers.
        start = bytes_data.find(b'\xff\xd8')
        end = bytes_data.find(b'\xff\xd9')
        if start != -1 and end != -1 and end > start:
            # Extract the JPEG image from the stream
            jpg = bytes_data[start:end+2]
            # Remove this image data from the buffer so it won't be processed again
            bytes_data = bytes_data[end+2:]
            
            # Generate a unique Unix timestamp (in milliseconds) for the filename.
            while True:
                ts = int(time.time() * 1000)
                filename = os.path.join(FRAME_DIR, f"{ts}.jpg")
                if not os.path.exists(filename):
                    save_frame(jpg, ts)
                    break
                time.sleep(0.001)  # Wait a millisecond before trying for a new timestamp
            
            frame_count += 1

            # Every batch_size frames, calculate and display the frame rate.
            if frame_count % batch_size == 0:
                current_time = time.time()
                elapsed = current_time - last_batch_time
                fps = batch_size / elapsed if elapsed > 0 else float('inf')
                print(f"Captured {frame_count} frames. Current FPS: {fps:.2f}")
                last_batch_time = current_time

            # Once we've captured max_frames, break from the loop.
            if frame_count >= max_frames:
                print("Captured 100 frames. Stopping.")
                break

    # Close the connection to stop the request.
    r.close()

if __name__ == "__main__":
    capture_frames(max_frames=300, batch_size=20)
