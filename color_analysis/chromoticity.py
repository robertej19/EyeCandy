import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image and convert to RGB
img = Image.open('data/frames/1744814236133.jpg').convert('RGB')
data = np.array(img)

# Convert to float and compute sum across RGB channels
data_float = data.astype(np.float32)
sum_channels = np.sum(data_float, axis=2, keepdims=True)

# Compute normalized chromaticity coordinates (avoid division by zero with a small constant)
rgb_norm = data_float / (sum_channels + 1e-6)

# Extract the normalized red and green channels
r = rgb_norm[:, :, 0].flatten()
g = rgb_norm[:, :, 1].flatten()
print("trying to plot")

# Scatter plot using the original color for each point
plt.figure(figsize=(6, 6))
plt.scatter(r, g, s=1, c=rgb_norm.reshape(-1, 3))
print("populated scatter")
plt.xlabel('Normalized Red (r)')
plt.ylabel('Normalized Green (g)')
plt.title('rg Chromaticity Diagram')
plt.grid(True)
print("saving pic")
#plt.show()
# save the figure
plt.savefig('rg_chromaticity_diagram.png', dpi=300, bbox_inches='tight')
print("saved pic")
plt.close()
print("closed pic")