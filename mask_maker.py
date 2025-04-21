import cv2
import numpy as np

# List to store points clicked by the user
points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f'Point added: ({x}, {y})')

        # Draw a small circle for visual feedback
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)

        # Draw lines between points if there is more than 1 point
        if len(points) > 1:
            cv2.line(img_display, points[-2], points[-1], (255, 0, 0), 2)

        cv2.imshow('Select Road Area', img_display)

# Load your reference frame
img = cv2.imread('test_frames/1745176127301.jpg')
img = cv2.rotate(img, cv2.ROTATE_180)
img_display = img.copy()

cv2.imshow('Select Road Area', img_display)
cv2.setMouseCallback('Select Road Area', click_event)

print("Click to outline the road. Press 'q' when done.")

# Wait until user presses 'q'
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Create and save the mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)
if len(points) >= 3:  # Need at least 3 points to form a polygon
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    cv2.imwrite('road_mask.png', mask)
    print('Mask saved as road_mask.png')
else:
    print('Error: You need at least 3 points to create a mask.')
