import os
# Set Qt to operate in offscreen mode (useful in headless environments)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
# Use a non-interactive backend for saving plots rather than displaying them
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import cv2

def load_image(image_path):
    """Load an image and convert it to an RGB numpy array."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def extract_dominant_colors(image, n_clusters=5):
    """Reshape image pixels and apply K-Means clustering to find dominant colors."""
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def rgb_to_lab(colors_rgb):
    """
    Convert an array of RGB colors to LAB.
    `colors_rgb` should be of shape (n, 3).
    """
    # OpenCV requires the input to be of shape (1, n, 3) and in 8-bit format.
    colors_rgb = np.uint8(colors_rgb.reshape(1, -1, 3))
    colors_lab = cv2.cvtColor(colors_rgb, cv2.COLOR_RGB2LAB)
    return colors_lab.reshape(-1, 3)

def lab_to_rgb(color_lab):
    """
    Convert a single LAB color (array of 3 values) back to RGB.
    Returns an array of shape (3,).
    """
    color_lab_uint8 = np.uint8(np.round(color_lab).reshape(1, 1, 3))
    color_rgb = cv2.cvtColor(color_lab_uint8, cv2.COLOR_LAB2RGB)
    return color_rgb[0, 0, :]

def generate_candidate_lab_grid(n_points=20):
    """
    Generate a grid of candidate colors in LAB space.
    OpenCV's LAB 8-bit space typically uses:
      L: [0, 255], a: [0, 255], b: [0, 255].
    The parameter `n_points` controls the resolution of the grid.
    """
    L_vals = np.linspace(0, 255, n_points)
    a_vals = np.linspace(0, 255, n_points)
    b_vals = np.linspace(0, 255, n_points)
    L, a, b = np.meshgrid(L_vals, a_vals, b_vals)
    candidate_grid = np.stack([L.flatten(), a.flatten(), b.flatten()], axis=1)
    return candidate_grid

def find_best_candidate(dominant_colors_lab, candidate_grid):
    """
    For each candidate color (in LAB), compute its minimum Euclidean distance 
    to the dominant colors. Return the candidate with the largest minimum distance.
    """
    diff = candidate_grid[:, None, :] - dominant_colors_lab[None, :, :]
    dist = np.linalg.norm(diff, axis=2)  # shape (N_candidates, N_dominant)
    min_distances = np.min(dist, axis=1)
    best_candidate_idx = np.argmax(min_distances)
    best_candidate_lab = candidate_grid[best_candidate_idx]
    return best_candidate_lab, min_distances[best_candidate_idx]

def save_dominant_colors_plot(dominant_colors_rgb, filename='dominant_colors.png'):
    """Save a palette image showing the dominant colors."""
    n = dominant_colors_rgb.shape[0]
    plt.figure(figsize=(n * 1.5, 2))
    for i, color in enumerate(dominant_colors_rgb):
        plt.subplot(1, n, i + 1)
        plt.imshow([[color]])
        plt.axis('off')
    plt.suptitle('Dominant Colors in the Image')
    plt.savefig(filename)
    plt.close()

def save_candidate_color_plot(best_candidate_rgb, filename='color_not_present.png'):
    """Save an image displaying the best candidate (the color not present)."""
    plt.figure(figsize=(2, 2))
    plt.imshow([[best_candidate_rgb]])
    plt.axis('off')
    plt.title('Color Not Present in Image')
    plt.savefig(filename)
    plt.close()

def main():
    # --- Parameters ---
    image_path = 'data/frames/1744814236133.jpg'  # Provided image path
    n_clusters = 5       # Number of dominant colors to extract
    grid_resolution = 20 # Controls candidate grid resolution in LAB space

    # --- Step 1: Load the Image ---
    image = load_image(image_path)

    # --- Step 2: Extract Dominant Colors via K-Means ---
    dominant_colors_rgb = extract_dominant_colors(image, n_clusters=n_clusters)
    print("Dominant Colors (RGB):")
    print(dominant_colors_rgb)
    
    # --- Save the Dominant Colors as a Palette ---
    save_dominant_colors_plot(dominant_colors_rgb, filename='dominant_colors.png')

    # --- Step 3: Convert Dominant Colors to LAB ---
    dominant_colors_lab = rgb_to_lab(dominant_colors_rgb)

    # --- Step 4: Generate Candidate Colors in LAB Space ---
    candidate_grid = generate_candidate_lab_grid(n_points=grid_resolution)

    # --- Step 5: Identify the Candidate Color Not Present ---
    best_candidate_lab, best_distance = find_best_candidate(dominant_colors_lab, candidate_grid)
    best_candidate_rgb = lab_to_rgb(best_candidate_lab)
    
    print("\nBest Candidate Color (LAB):", best_candidate_lab)
    print("Best Candidate Color (RGB):", best_candidate_rgb)
    print("Distance in LAB space from nearest dominant color:", best_distance)

    # --- Save the Best Candidate Color ---
    save_candidate_color_plot(best_candidate_rgb, filename='color_not_present.png')

if __name__ == '__main__':
    main()
