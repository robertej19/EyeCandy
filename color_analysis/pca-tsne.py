import os
# -- headless setup (no Qt) --
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_image(path):
    """Load and return an RGB image as an (H*W, 3) array of pixels."""
    img = Image.open(path).convert('RGB')
    data = np.array(img)
    h, w, _ = data.shape
    return data.reshape(-1, 3), (h, w)

def cluster_pixels(pixels, n_clusters=5):
    """Fit KMeans and return labels and cluster centers."""
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_
    return labels, centers

def subsample(pixels, labels, max_samples=20000):
    """Randomly subsample pixels and their labels for faster projection."""
    N = pixels.shape[0]
    idx = np.random.choice(N, size=min(max_samples, N), replace=False)
    return pixels[idx], labels[idx]

def plot_and_save(proj2d, labels, title, fname):
    """Scatter proj2d colored by labels, then save to fname."""
    plt.figure(figsize=(6,6))
    plt.scatter(proj2d[:,0], proj2d[:,1], s=1, c=labels, cmap='tab10')
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def main():
    image_path    = 'data/frames/1744814236133.jpg'
    n_clusters    = 5
    subsample_max = 20000

    # 1) Load & reshape
    pixels, (H, W) = load_image(image_path)

    # 2) Cluster
    labels, centers = cluster_pixels(pixels, n_clusters=n_clusters)

    # 3) Subsample for projection
    pix_small, lbl_small = subsample(pixels, labels, max_samples=subsample_max)

    # 4) PCA projection
    pca = PCA(n_components=2, random_state=42)
    proj_pca = pca.fit_transform(pix_small)
    plot_and_save(proj_pca, lbl_small,
                  title='PCA of Pixels – Colored by Cluster',
                  fname='pca_clusters.png')

    # 5) t-SNE projection
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    proj_tsne = tsne.fit_transform(pix_small)
    plot_and_save(proj_tsne, lbl_small,
                  title='t‑SNE of Pixels – Colored by Cluster',
                  fname='tsne_clusters.png')

if __name__ == '__main__':
    main()
