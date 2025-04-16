import requests
import os
import time

# URL of your MJPEG stream
STREAM_URL = "http://192.168.1.190:8485/video"

# Create a unique directory at startup using the current Unix timestamp in milliseconds.
start_timestamp = int(time.time() * 1000)
FRAME_DIR = f"frames_{start_timestamp}"
os.makedirs(FRAME_DIR, exist_ok=True)
print(f"Saving frames to directory: {FRAME_DIR}")

def save_frame(jpg_bytes, timestamp):
    """Save the JPEG frame to disk using the provided timestamp as the filename."""
    filename = os.path.join(FRAME_DIR, f"{timestamp}.jpg")
    with open(filename, "wb") as f:
        f.write(jpg_bytes)

def capture_frames(batch_size=20):
    # Open the MJPEG stream.
    r = requests.get(STREAM_URL, stream=True)
    bytes_data = b""
    frame_count = 0

    # Initialize the batch timer for framerate calculation.
    last_batch_time = time.time()
    
    try:
        for chunk in r.iter_content(chunk_size=1024):
            bytes_data += chunk

            # Look for the JPEG start and end markers.
            start = bytes_data.find(b'\xff\xd8')
            end = bytes_data.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                # Extract the JPEG frame.
                jpg = bytes_data[start:end+2]
                # Remove the processed frame from the buffer.
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
    except KeyboardInterrupt:
        print("Capture interrupted by user.")
    finally:
        r.close()
        print("Stopped frame capture.")

if __name__ == "__main__":
    capture_frames(batch_size=200)
