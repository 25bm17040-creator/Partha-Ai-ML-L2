import cv2
import numpy as np

image=cv2.imread("sample.jpg")
greyimage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  

sobel_X=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_Y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])


output_X = cv2.Sobel(greyimage, cv2.CV_64F, 1, 0, ksize=3)
output_Y = cv2.Sobel(greyimage, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined=cv2.magnitude(output_X.astype(np.float32),output_Y.astype(np.float32))
sobel_combined = cv2.normalize(
    sobel_combined,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)
sobel_combined = sobel_combined.astype(np.uint8)

print(output_X)
print(output_Y)
print(sobel_combined)
#cv2.imshow("Original Image",image)
#cv2.imshow("Grey Image",greyimage)
#cv2.imshow("output_X",output_X)
#cv2.imshow("output_Y",output_Y)
#cv2.imshow("output",cv2.convertScaleAbs(sobel_combined))
cv2.imwrite("edges.png", sobel_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()