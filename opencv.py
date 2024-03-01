import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('ob.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray,235,255,cv.THRESH_BINARY_INV)
contours, _, = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours,4, (0,0,255),3)
cv.imshow('edged', img)
cv.waitKey(0)
cv.destroyAllWindows()