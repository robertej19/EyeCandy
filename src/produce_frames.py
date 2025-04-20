from flask import Flask, Response
import cv2
from picamera2 import Picamera2


app = Flask(__name__)

def gen_frames():
    # Initialize the camera and configure the preview resolution.
    picam2 = Picamera2()
    #config = picam2.create_preview_configuration({"size": (1280,720)})
    config = picam2.create_preview_configuration({"size": (4056,3040)})
    #config = picam2.create_video_configuration({"size": (2028,1080)})
    #config = picam2.create_video_configuration({"size": (2028,1520)})
    picam2.configure(config)
    picam2.start()  # Start camera capture for preview

    while True:
        # Capture a frame as a NumPy array
        frame = picam2.capture_array()
        if frame is None:
            continue

        #crop frame to be from half of x range and all of the bottom 2/3 of y range
        frame = frame[0:3040, 0:2100]
        


        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # The server listens on all interfaces. Replace '8485' with any port you prefer.
    app.run(host='0.0.0.0', port=8485, threaded=True)


































