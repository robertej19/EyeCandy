import os
import glob
import cv2
import numpy as np

def compute_median_background(frame_dir: str,
                              pattern: str = "*.*",
                              save_to: str | None = None) -> np.ndarray:
    """
    Build a static‑background estimate by taking the per‑pixel median
    across all frames in `frame_dir`.

    Parameters
    ----------
    frame_dir : str
        Directory containing road frames (e.g. "test_frames/").
    pattern   : str, default "*.*"
        Glob pattern for frame filenames ("*.jpg", "*.png", etc.).
    save_to   : str | None, default None
        If given, the median image is written to this path.

    Returns
    -------
    median_img : np.ndarray (H×W×3, uint8)
        The background image.
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No images matching {pattern} in {frame_dir}")

    # only invlude every 20th frame
    paths = paths[::20]
    # Stack frames into one big array (N, H, W, C)
    frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    frames = [f for f in frames if f is not None]  # drop unreadables
    median_img = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    if save_to:
        cv2.imwrite(save_to, median_img)

    return median_img

# Example usage
compute_median_background("data/medium_test_2/", pattern="*.jpg", save_to="second_median_bg.jpg")
