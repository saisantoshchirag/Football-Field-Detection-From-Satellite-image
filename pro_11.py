import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(len(flags))
print(flags[40])


#brg to rgb
img2 = cv2.imread('2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img = cv2.medianBlur(img2, 35)
imgb = cv2.bilateralFilter(img2, 5, 75, 75)



#printing the image
plt.imshow(img2)
print("the image")
plt.show()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(imgb)
plt.show()

#rgb to hsv
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_imgb = cv2.cvtColor(imgb, cv2.COLOR_RGB2HSV)
print("median and biateral hsv comparision")
plt.subplot(1, 2, 1)
plt.imshow(hsv_img)
plt.subplot(1, 2, 2)
plt.imshow(hsv_imgb)
plt.show()



light_green = (40, 80, 40)
dark_green = (80, 255, 255)
light_green1 = (40, 40, 40)
dark_green1 = (70, 255, 255)


#showing the range
lo_square = np.full((10, 10, 3), light_green, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_green, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
print("dark green, light green")
plt.show()


#create a mask
mask2 = cv2.inRange(hsv_img, light_green, dark_green)
mask1 = cv2.inRange(hsv_img, light_green1, dark_green1)
mask = mask2+mask1

mask2b = cv2.inRange(hsv_imgb, light_green, dark_green)
mask1b = cv2.inRange(hsv_imgb, light_green1, dark_green1)
maskb = mask2b+mask1b

#impose the mask
result =  cv2.bitwise_and(img, img, mask=mask)
img=imgb
cv2.imwrite("mask.jpg", mask)

#print it
plt.subplot(1, 4, 1)
plt.imshow(mask)
#plt.imshow(mask, cmap="gray")
mask=maskb
resultb = resultb = cv2.bitwise_and(img, img, mask=mask)
plt.subplot(1, 4, 2)
plt.imshow(result)
plt.subplot(1, 4, 3)
plt.imshow(maskb)
plt.subplot(1, 4, 4)
plt.imshow(resultb)
plt.show()


#here the mask has some impurities in it. we need to remove that first.

cv2.imwrite("maskb.jpg", maskb)

image = cv2.imread('mask.jpg')
imageb = cv2.imread('maskb.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageb = cv2.cvtColor(imageb, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(image, 45)
medianb = cv2.medianBlur(imageb, 45)
cv2.imwrite('out1.png',image)




def contour(image):
    img = cv2.imread(image)
    t = 10

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    (t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)

    (_, contours, _) = cv2.findContours(binary, cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_SIMPLE)
    # contours if for all the contours of the image
    print("Found %d objects." % len(contours))
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))
    # contours length - length of all contours
    contours_length = [len(c) for i, c in enumerate(contours)]
    contours_length.sort()
    # contours temp - length of contours of top 30 percent contours
    contours_temp = [contours_length[i] for i in range(int(len(contours_length) * 0.70), len(contours_length))]
    contours_image = []
    count = 0
    for i in contours:
        if len(i) in contours_temp:
            contours_image.append(i)
            count += 1

    contours_image_len = [len(i) for i in contours_image]
    print('\ncsd\n', contours_image_len)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 15)

    return img

#edit the filename based on the inpput name of the file
filename = 'out1.png'
t = 10

image1 = contour(filename)
cv2.imwrite("out2.png",image1)





imageb = cv2.imread('out2.png')
imageb = cv2.cvtColor(imageb, cv2.COLOR_BGR2GRAY)

#identify rect

#ayush using chirag output
medianb = cv2.medianBlur(imageb, 15)
#medium blur to chirag, up
cv2.imwrite("ayush-chirag.jpg", medianb)
print("median blur to chirag output")

#saurabh using ayush output

image_gray = cv2.imread("ayush-chirag.jpg", 0)
image_gray = np.where(image_gray > 30, 255, image_gray)
image_gray = np.where(image_gray <= 30, 0, image_gray)

_, contours, _ = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rect_cnts = []
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 4: # shape filtering condition
        rect_cnts.append(cnt)

max_area = 0
football_square = None
for cnt in rect_cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    if max_area < w*h:
        max_area = w*h
        football_square = cnt

# Draw the result
image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
cv2.drawContours(image, [football_square], -1, (0, 0,255), 5)

cv2.imwrite("output.jpg", image)
cv2.imshow("Result Preview", image)
cv2.waitKey()



