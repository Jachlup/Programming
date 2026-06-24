import cv2

image = cv2.imread("test_image.jpg")
cv2.imshow("Test Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()