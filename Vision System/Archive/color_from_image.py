import cv2
import numpy as np

img = cv2.imread("Photos/Finger1_deflected.jpg")
img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Yellow range in HSV (single continuous hue range)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Create mask for yellow
full_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Remove small noise by 'opening' (erosion then dilation)
kernel = np.ones((5, 5), np.uint8)
clean_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)



# 1. Find the boundaries (contours)
contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # 2. Calculate the "Moments" (mathematical center of the shape)
    M = cv2.moments(cnt)
    
    if M["m00"] != 0: # Avoid division by zero
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # 3. Print the position and draw a circle on the original image
        print(f"Yellow dot found at: X={cX}, Y={cY}")
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1) # Draw a white dot at the center
        cv2.putText(img, "point", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)




# Show the results
cv2.imshow("full mask", full_mask)
cv2.imshow("Clean mask", clean_mask)
cv2.imshow("Detected Dots", img)
cv2.waitKey(0)
cv2.destroyAllWindows()