import tracemalloc
from skimage.metrics import structural_similarity as ssim
import cv2

tracemalloc.start()

# Load the two images to compare
image1 = cv2.imread('img/circle1.png', cv2.IMREAD_GRAYSCALE)  # Load as grayscale
image2 = cv2.imread('img/square.png', cv2.IMREAD_GRAYSCALE)  # Load as grayscale

# Compute SSIM
score, diff = ssim(image1, image2, full=True)

print(f"SSIM score: {score}")

print(tracemalloc.get_traced_memory())
tracemalloc.stop()
