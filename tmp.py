'''
Assumptions:
* Binarized cleaned image
* Preliminary OCR available

Desired Outcomes:
* Detect tables with lines (excel)
* Detect tables without lines
* Split tables into rows and columns
* Detect horizontal rules
* Detect boxes
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy import ndimage

# import warnings
# import scipy.ndimage.filters as filters # median etc


def get_image(*args, **kwargs):
    pass

filename = 'out/bokslut-001.png'
filename = 'data/examples-30.png'
filename = 'data/examples-6.png'

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(image=edges, rho=2, theta=np.pi/180,
                        threshold=10, minLineLength=50, maxLineGap=10)
a,b,c = lines.shape
print('Hough:', lines.shape)


ret, thresh = cv2.threshold(gray,125,255,0)
thresh = (255-thresh)
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3) #draw all contours
plt.imshow(img)
plt.show(block=False)


lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)


img = 255 - cv2.imread(filename, cv2.IMREAD_COLOR)
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)
print('size:', size, img.shape)

ret, img = cv2.threshold(img, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
done = False

while not done:
#for i in range(500):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True
    print('z', zeros)

plt.imshow("skeleton", skel)




filename = 'data/examples-30.png'

img = cv2.imread(filename, cv2.IMREAD_COLOR)

rows, cols, _ = img.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
plt.imshow(dst)
plt.show()
img = dst


# perform skeletonization

gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(gray, 127, 255, 0)[1] // 255
skeleton = skeletonize(image)
plt.imshow(skeleton)
plt.show()




element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
eroded = cv2.erode(image, element, (-1, -1))
dilated = cv2.dilate(eroded, element, (-1, -1))
img_pp = dilated.copy()
plt.imshow(img_pp)
plt.show()


gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(gray, 127, 255, 0)[1] // 255

edges = filters.sobel_h(image) > 0
plt.imshow(edges)
plt.show()


img = np.array(skeleton*255, dtype=np.uint8)
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
#sobely=sobelx

img = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(gray, 127, 255, 0)[1] // 255
skeleton = skeletonize(image)

img = np.array(skeleton*255, dtype=np.uint8)
# sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
thresh = img # sobelx

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3) #draw all contours
plt.imshow(img)
plt.show(block=False)


minLineLength = 100
# minLineLength = 1500
maxLineGap = 5
lines = cv2.HoughLinesP(sobely,2,np.pi/180,threshold=200,minLineLength=minLineLength,maxLineGap=maxLineGap)
print('lines =', lines.shape)
img = dst.copy()
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
plt.imshow(img)
plt.show()


# lines.reshape((-1, 4))
import math
l = []
k = []
for x1,y1,x2,y2 in lines.reshape((-1, 4)):
    dx=float(x2-x1)
    dy=float(y2-y1)
    k.append(dx/dy)
    l.append(math.sqrt(dx*dx+dy*dy))
k = np.array(k)
l = np.array(l)
avg_slope = k.mean()


## MAY WLOG ASSUME that lines are almost horizontal (vertical)
## Hence, y-coords are similar and we can partition based on y ...

a = np.hstack((q, k.reshape((-1, 1)))).astype(int)
a[a[:,1].argsort()]

q = lines.reshape((-1, 4))
q[np.lexsort(lines.T, axis=1)]


q[q[:,1].argsort()]


image = cv2.threshold(gray, 127, 255, 0)[1]
med = ndimage.median_filter(image, size=(1, 51))
#plt.imshow(cv2.bitwise_and(med, img))
plt.imshow(med)
plt.show(block=False)


## https://stackoverflow.com/questions/22745076/libpng-warning-iccp-known-incorrect-srgb-profile

## image is binary image with bg = 0 and fg = 1
## TODO: NOT GAP ... opposite of gap ... "distance to background"
def get_horizontal_gap_image(binary_image):
    rows, cols = binary_image.shape
    forward = np.zeros(binary_image.shape, dtype=np.uint16)
    backward = np.zeros(binary_image.shape, dtype=np.uint16)
    
    for i in range(cols):
        forward[:, i] = (forward[:, i-1] + 1) * binary_image[:, i]
        backward[:, -(i+1)] = (backward[:, -i] + 1) * binary_image[:, -(i+1)]
    return (forward + backward)


def get_vertical_gap_image(binary_image):
    return get_horizontal_gap_image(binary_image.T).T


## image: fg = 1, bg = 0
def get_fg_mask(binary_image, h_tresh, v_tresh):
    inv_image = np.logical_not(binary_image)
    h_img = (get_horizontal_gap_image(inv_image) < h_tresh)
    v_img = (get_horizontal_gap_image(inv_image.T).T < v_tresh)
    result = np.logical_or(h_img, v_img)
    return result


## image: fg = 1, bg = 0
def get_line_mask(binary_image, h_tresh, v_tresh):
    h_img = (get_horizontal_gap_image(binary_image) > h_tresh)
    v_img = (get_horizontal_gap_image(binary_image.T).T > v_tresh)
    result = np.logical_or(h_img, v_img)
    return result

## report line ... report grid


filename = 'data/examples-30.png'
filename = 'data/examples-15.png'
filename = 'data/examples-7.png'
filename = 'data/examples-6.png'

img = cv2.imread(filename)
gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_image = cv2.threshold(gray, 127, 255, 0)[1] // 255

c_img = 255 * np.ones(img.shape, dtype=np.uint8)
h_tresh, v_tresh = 250, 3 # 150, 100
c_img[:,:,0] = 255 * get_line_mask(binary_image, h_tresh, v_tresh).astype(np.uint8)
#c_img[:,:,1] = 255 * get_fg_mask(binary_image, h_tresh, v_tresh).astype(np.uint8)
c_img[img == 0] = 0
c_img[:, :, 0] = (get_horizontal_gap_image(binary_image) > 100).astype(np.uint8) * 255
plt.imshow(c_img, 'gray')
plt.show(block=False)




## median (3, 3) for "outside" == keepers?

filename = 'data/examples-7.png'

img = cv2.imread(filename)
gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_image = cv2.threshold(gray, 127, 255, 0)[1] // 255

c_img = img.copy()
t_img = gray.copy() * 0
h_mask = (get_horizontal_gap_image(binary_image) > 100) # .astype(np.uint8) # * 255
im2, contours, hierarchy = cv2.findContours(h_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(t_img, contours, -1, 255, 7) # 5 or 7 ?
cv2.drawContours(t_img, contours, -1, 127, 3)
t_img[h_mask] = 0
c_img[np.logical_and(t_img == 127, c_img[:, :, 0] == 255)] = 90   # 
c_img[np.logical_and(t_img == 127, c_img[:, :, 0] == 0)] = 60     # Inner=Thin contour and image black
c_img[np.logical_and(t_img == 255, c_img[:, :, 0] == 255)] = 150  # 
c_img[np.logical_and(t_img == 255, c_img[:, :, 0] == 0)] = 120    # Outer=Fat contour and image black
c_img[h_mask] = 200

plt.imshow(c_img)
plt.show(block=False)


# debug
c_img[t_img == 0] = 80
c_img[t_img == 255] = 220

##




filename = 'data/examples-30.png'
filename = 'data/examples-15.png'
filename = 'data/examples-7.png'
filename = 'data/examples-6.png'
img = cv2.imread(filename)
gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_image = cv2.threshold(gray, 127, 255, 0)[1] // 255

c_img = 255 * np.ones(img.shape, dtype=np.uint8)
h_tresh, v_tresh = 30, 30 # 150, 100
#c_img[:,:,0] = 255 * get_line_mask(binary_image, h_tresh, v_tresh).astype(np.uint8)
c_img[:,:,1] = 255 * get_fg_mask(binary_image, h_tresh, v_tresh).astype(np.uint8)
c_img[img == 0] = 0
plt.imshow(c_img)
plt.show(block=False)


# plt.imshow(image, 'gray'); plt.show(block=False)

tresholds = [ (30, 30), (60, 60), (120, 30), (120, 30), ]
tresholds = [ (i, i) for i in range(30, 600, 20)]
#
g_img = np.ones(img.shape, dtype=np.uint8)
inv_image = np.logical_not(binary_image)
h_img = get_horizontal_gap_image(inv_image)
v_img = get_horizontal_gap_image(inv_image.T).T
for idx, (h_tresh, v_tresh) in enumerate(reversed(tresholds)):
    value = (255 * (idx + 1) / len(tresholds))
    result = np.logical_or(h_img < h_tresh, v_img < v_tresh)
    g_img[result] = value
plt.imshow(g_img, 'gray') ; plt.show(block=False)
    

# inv_image = np.logical_not(binary_image)
# h_img = get_horizontal_gap_image(inv_image)
# v_img = get_horizontal_gap_image(inv_image.T).T
# x_img = ((h_img / 25) * (v_img / 18)).astype(np.uint8)
# plt.imshow(v_img, 'gray') ; plt.show(block=False)
# plt.imshow(np.hstack([h_img, v_img]), 'gray') ; plt.show(block=False)

    
# vis = x_img.copy()
# mser = cv2.MSER_create()
# regions = mser.detectRegions(img, None)
# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
# plt.imshow(vis)
# plt.show(block=False)




filename = 'data/examples-6.png'
# filename = 'data/examples-30.png'
# filename = 'data/examples-23.png'
# filename = 'data/examples-15.png'
img = cv2.imread(filename)
# gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(gray, 127, 255, 0)[1] // 255

h_tresh, v_tresh = 100, 30
h_img = (get_horizontal_gap_image(image) > h_tresh)
v_img = (get_horizontal_gap_image(image.T).T > v_tresh)
x_img = (h_img * v_img * 255).astype(np.uint8)
# plt.imshow(np.hstack([x_img, gray]), 'gray')
# plt.imshow(x_img, 'gray')

gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image = cv2.threshold(gray, 127, 255, 0)[1] // 255
l_img = ((get_horizontal_gap_image(image) > 400) * 255).astype(np.uint8)
l_img = ((get_horizontal_gap_image(image.T).T > 200) * 255).astype(np.uint8)

c_img = img.copy() * 0
# c_img[:,:,0] = 255 - img[:,:, 0]
#c_img[:,:,1] = 255 - x_img  # x_img is 255 for empty areas
c_img[:,:,2] = l_img  # l_img is 255 for horizontal lines
# plt.imshow(255 - c_img)
plt.imshow(c_img)
plt.show(block=False)


plt.imshow(image); plt.show(block=False)


out = (get_horizontal_gap_image(image) / 10).astype(np.uint8)
plt.imshow(np.hstack([out.T, gray.T]), 'gray')
#plt.imshow(np.hstack([out, gray]), 'gray')
#plt.imshow(out, 'gray')
plt.show(block=False)

img[:, :, 0] = out.T
plt.imshow(img)
plt.show(block=False)

out = ((out > 300) * 255).astype(np.uint8)
im2, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_c = img.copy()
cv2.drawContours(img_c, contours, -1, (0,255,0), 1) #draw all contours

plt.imshow(img_c)
plt.show(block=False)




element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)
cv2.drawContours(opening, contours, -1, (0,255,0), 1) #draw all contours
plt.imshow(opening)
plt.show(block=False)


contours[0].reshape((-1, 2)) # fit line ...


plt.imshow(out, 'gray')
plt.show(block=False)
