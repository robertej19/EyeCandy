import cv2
import numpy as np
import os
import glob

def detect_multiscale_blobs(
    fgmask: np.ndarray,
    cuts: list[float] = [0.25, 0.75],
    min_areas: list[int] = [10, 200, 1000],
    kernels: list[np.ndarray] = None
) -> list[tuple[int,int,int,int]]:
    """
    fgmask   : binary foreground mask (H×W)
    cuts     : either fractions of H (e.g. [0.25,0.75]) or absolute Y pixels ([y1,y2])
    min_areas: minimum w*h in each band
    kernels  : list of 3 structuring elements (one per band)
    """
    H, W = fgmask.shape[:2]

    # ——— interpret cuts as fractions if <1.0 ———
    if any(c <= 1.0 for c in cuts):
        cuts_px = [int(c * H) for c in cuts]
    else:
        cuts_px = cuts
    cuts_px = sorted(cuts_px)

    # ——— default morphology kernels if none passed ———
    if kernels is None:
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
        ]

    detections = []

    # process each band separately
    bands = [ (0, cuts_px[0]),
              (cuts_px[0], cuts_px[1]),
              (cuts_px[1], H) ]



    for band_idx, (y0, y1) in enumerate(bands):
        roi = fgmask[y0:y1, :]
        # clean up the mask in this band
        #cleaned = cv2.morphologyEx(roi, cv2.MORPH_OPEN,  kernels[band_idx], iterations=1)
        #cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernels[band_idx], iterations=2)


        # inside your per‐band loop, before findContours():
        eroded = cv2.erode(roi, kernels[band_idx], iterations=1)
        restored = cv2.dilate(eroded, kernels[band_idx], iterations=1)



        # find contours in the cleaned band
        cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y_rel, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_areas[band_idx]:
                continue

            # convert y back to full‐image coords
            y = y0 + y_rel
            detections.append((x, y, w, h))

    return detections


def process_frame(frame):
    # load median bg
    median_bg = cv2.imread('second_median_bg.jpg')
    # rotate the median bg
    median_bg = cv2.rotate(median_bg, cv2.ROTATE_180)
    road_mask = cv2.imread("micro_test_files/road_mask.png")
    h,w = road_mask.shape[:2]
    masked_image = cv2.bitwise_and(frame, road_mask)
    masked_bg = cv2.bitwise_and(median_bg, road_mask)

    threshold_line = [int(0.3*(h+250)), int(0.6*(w+250))] 


    diff = cv2.absdiff(masked_image, masked_bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fgmask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)


    dets = detect_multiscale_blobs(fgmask, cuts=threshold_line)
    output_image = frame.copy()
    # draw rectangles around the detected blobs
    for (x,y,w,h) in dets:
        cv2.rectangle(output_image, (x,y), (x+w,y+h), (0,0,255), 2)

    # return the processed image
    return output_image, fgmask, masked_image, masked_bg

if __name__ == "__main__":
    # load the input image

    # create an output directory for processed images
    output_dir = "test_output_frames_2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    paths = sorted(glob.glob(os.path.join("data/medium_test_2/", "*.jpg")))


    for p in paths:
        print(p)
        frame = cv2.imread(p, cv2.IMREAD_COLOR)
        #frame = cv2.imread("micro_test_files/1745176127617.jpg")

        # camera is upside down by default
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # process the image
        output_image, fgmask, masked_image, masked_bg = process_frame(frame)

        # save the processed image, name as original_name_processed.jpg
        original_name = os.path.basename(p)
        #print(original_name)
        output_name = os.path.join(output_dir, original_name.replace(".jpg", "_processed.jpg"))
        cv2.imwrite(output_name, output_image)