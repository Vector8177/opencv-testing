import cv2
import numpy as np


def remove_similar_tuples(arr, threshold):
    result = []
    # iterate through each tuple in the array
    for tup in arr:
        # assume the tuple will not be removed by default
        keepr = True
        # iterate through the result list to check if any tuples are too similar to the current tuple
        for res_tup in result:
            keep = 0
            # compare each element of the tuples to see if they are within the threshold
            for i in range(2):
                if abs(tup[i] - res_tup[i]) < threshold:
                    # if any element is too different, mark the tuple for removal and break the inner loop
                    keep += 1

            # if the tuple is marked for removal, break the outer loop and move on to the next tuple
            if keep > 1:
                keepr = False
                break

        # if the tuple was not marked for removal, add it to the result list
        if keepr:
            result.append(tup)

    # return the modified result list
    return result


# Load image and resize
img = cv2.imread('cone.png')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur image to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Define yellow color range in HSV
lower_yellow = np.array([18, 112, 130])
upper_yellow = np.array([96, 255, 255])

# Convert image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold the image to get only yellow colors
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Erode and dilate the mask to remove small blobs and fill gaps
kernel = np.ones((5, 5), np.uint8)
mask = cv2.Canny(mask, 175, 200)

cv2.imshow('Masked image', mask)

# Detect lines using Hough transform on the blurred image
lines = cv2.HoughLines(mask, rho=2, theta=np.pi/180,
                       threshold=65)
intersections = []
# Find intersection points and angles between lines
for i in range(len(lines)):
    rho1, theta1 = lines[i][0]
    for j in range(i+1, len(lines)):
        rho2, theta2 = lines[j][0]
        if abs(theta1 - theta2) > 0.1:  # check for non-parallel lines
            A = np.array([[np.cos(theta1), np.sin(theta1)],
                         [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            try:
                x0, y0 = np.linalg.solve(A, b)
                x0, y0 = int(x0), int(y0)
                angle = np.abs(theta1 - theta2) * 180 / np.pi
                intersections.append(tuple((x0, y0, angle)))

            except np.linalg.LinAlgError:
                continue  # skip parallel lines

# Display the lines on the original image
intersections = remove_similar_tuples(intersections, 20)
for intersection in intersections:
    print(
        f"Intersection at ({intersection[0]}, {intersection[1]}) with angle {intersection[2]:.2f} degrees")
    cv2.circle(img, (intersection[0], intersection[1]), 20, (0, 0, 255), 10)
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Lines and intersections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
