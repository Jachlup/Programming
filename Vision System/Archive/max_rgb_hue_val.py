import cv2
import numpy as np

img = cv2.imread("Photos/color_check.jpg")

img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)

img_blue, img_red, img_green = cv2.split(img)


(minValB, maxValB, minLocB, maxLocB) = cv2.minMaxLoc(img_blue)
(minValG, maxValG, minLocG, maxLocG) = cv2.minMaxLoc(img_green)
(minValR, maxValR, minLocR, maxLocR) = cv2.minMaxLoc(img_red)

# Get RGB values at max points
blue_rgb = img[maxLocB [1], maxLocB [0]]
green_rgb = img[maxLocG[1], maxLocG[0]]
red_rgb = img[maxLocR[1], maxLocR[0]]

print(f"Blue max point at {maxLocB }: RGB = {blue_rgb}, Max Value = {maxValB}")
print(f"Green max point at {maxLocG}: RGB = {green_rgb}, Max Value = {maxValG}")
print(f"Red max point at {maxLocR}: RGB = {red_rgb}, Max Value = {maxValR}")

cv2.circle(img, (maxLocB   ), 7, (255, 255, 255), -1) # Draw a white dot at the center
cv2.putText(img, "blue", (maxLocB  [0] - 20, maxLocB [1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.circle(img, (maxLocG), 7, (255, 255, 255), -1) # Draw a white dot at the center
cv2.putText(img, "green", (maxLocG[0] - 20  , maxLocG[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.circle(img, (maxLocR), 7, (255, 255, 255), -1) # Draw a white dot at the center
cv2.putText(img, "red", (maxLocR[0] - 20, maxLocR[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



cv2.imshow('original', img)
cv2.imshow('blue', img_blue)
cv2.imshow('red', img_red)
cv2.imshow('green', img_green)
cv2.waitKey(0)
cv2.destroyAllWindows()