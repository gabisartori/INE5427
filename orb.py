import tracemalloc
import cv2

# Começar a rastrear a alocação de memória
tracemalloc.start()

# Read the query image as query_img
# and train image This query image
# is what you need to find in train image
# Save it in the same directory
# with the name image.jpg  
train_img = cv2.imread('img/circle1.png')
query_img = cv2.imread('img/square.png')
 
# Convert it to grayscale
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
 
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
 
# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)

# Initialize the Matcher for matching
# the keypoints and then match the
# keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors,trainDescriptors)

# Encerrar o rastreamento da alocação de memória
print(tracemalloc.get_traced_memory())
tracemalloc.stop()

# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:20],None)
print(len(queryKeypoints))
