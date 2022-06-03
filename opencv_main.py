# pip install opencv-python
# conda install opencv-puython


import cv2

image_path = "C:/Users/goura/OneDrive/Pictures/Photo-1.jpeg"

image = cv2.imread(image_path )
# (0 , 255)


cv2.rectangle(image, (200, 300), (500, 500), (0, 255, 0), 10)
cv2.circle(image, (200, 300), 50, (0, 0, 255), 10)

cv2.imshow("Image", image)
cv2.waitKey(0)
