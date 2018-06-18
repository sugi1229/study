import cv2
import math

img = cv2.imread('08.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#エッジ処理
edges = cv2.Canny(gray,50,150,apertureSize = 3)

#ハフ変換
lines = cv2.HoughLinesP(edges, 1, math.pi/180.0, 50, 30, 5)
# lines = cv2.HoughLines(edges, 1, math.pi/180.0, 50)

if lines is not None:
    for (x1, y1, x2, y2) in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imwrite('result.png', img)
