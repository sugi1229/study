import matplotlib
matplotlib.use('Agg')
import cv2
from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns

img = cv2.imread('DV120313_00830 2 2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.ones((2,2),np.float32)/4
dst = cv2.filter2D(gray,-1,kernel)
#dst = cv2.filter2D(dst,-1,kernel)
cv2.imwrite("dst.png",dst)

img = dst
#img = cv2.imread('LennaG2.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ラプラシアン
dst3 = cv2.Laplacian(img, cv2.CV_32F, ksize=5)
cv2.imwrite("output3-5.jpg", dst3)

dst3 = -1*dst3
dst3[dst3>=-800]=1500
#sns.heatmap(dst3)

#dst3[dst3>=0]=0
#dst3[dst3<0]*=-1

cv2.imwrite("output4-5.jpg", dst3)

#print(dst3)

num = dst3.shape[1]
y = np.linspace(0,1,num)
Ym = np.sum(dst3,axis=0)

yy = np.mean(Ym[:(num-10)])
print(yy)


num = img.shape[0]
x = np.linspace(0,1,num)
Xm = np.sum(dst3,axis=1)

plt.plot(y,Ym)
#plt.plot(x,Xm)
plt.savefig("test-5.png")

