import cv2
import matplotlib.pyplot as plt
import tracemalloc

tracemalloc.start()
# Load the images to be compared
image1 = cv2.imread('img/circle1.png', cv2.IMREAD_GRAYSCALE)  # Query image
image2 = cv2.imread('img/circle2.png', cv2.IMREAD_GRAYSCALE)  # Train image

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Use BFMatcher to match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

print(tracemalloc.get_traced_memory())
tracemalloc.stop()

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(matched_image)
plt.title('Feature Matches')
plt.axis('off')
plt.show()