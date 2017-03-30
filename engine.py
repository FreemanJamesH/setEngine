import cv2
import numpy as np
import math

game = cv2.imread('home1cleaner.jpg')
image_area =  game.shape[0] * game.shape[1]

diamond = cv2.imread('shapeTemplates/diamond.jpg')
oval = cv2.imread('shapeTemplates/oval.jpg')
squiggle = cv2.imread('shapeTemplates/squiggle.jpg')

ret, diamondThresh = cv2.threshold(cv2.imread('shapeTemplates/diamond.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
ret, ovalThresh = cv2.threshold(cv2.imread('shapeTemplates/oval.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
ret, squiggleThresh = cv2.threshold(cv2.imread('shapeTemplates/squiggle.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
diamondContours, hierarchy = cv2.findContours(diamondThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
ovalContours, hierarchy = cv2.findContours(ovalThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
squiggleContours, hierarchy = cv2.findContours(squiggleThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

def pythagorean(distanceXY):
    return ((distanceXY[0] ** 2) + (distanceXY[1] ** 2))


contours = []
cards = []

gamegray = cv2.cvtColor(game, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gamegray, 150, 255, 0)
kernel = np.ones((4,4), np.uint8)

eroded = cv2.erode(thresh, kernel, iterations = 1)

contoursDirty, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contoursDirty:
    area = cv2.contourArea(cnt)
    if (area/image_area > 0.01):
        contours.append(cnt)

cv2.drawContours(game, contours, -1, (0,255,0), 3)


cv2.imshow('game w/ contours', game)
#
print "contours found:", len(contours)


# preprocessing

for (i, c) in enumerate(contours):
    quad1 = []
    quad2 = []
    quad3 = []
    quad4 = []

    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    for (i, d) in enumerate(c):

        distance = [(d[0][0] - cx), (d[0][1] - cy)]
        if (distance[0] < 0) and (distance[1] < 0) :
            quad1.append(distance)
        if (distance[0] < 0) and (distance[1] > 0) :
            quad2.append(distance)
        if (distance[0] > 0) and (distance[1] > 0) :
            quad3.append(distance)
        if (distance[0] > 0) and (distance[1] < 0) :
            quad4.append(distance)

    top_left = sorted(quad1, key = pythagorean, reverse = True)[0]
    bottom_left = sorted(quad2, key = pythagorean, reverse = True)[0]
    bottom_right = sorted(quad3, key = pythagorean, reverse = True)[0]
    top_right = sorted(quad4, key = pythagorean, reverse = True)[0]

    distance_left_midpoint_above_centroid = (top_left[1] + bottom_left[1])/-2
    distance_left_midpoint_from_center = (top_left[0] + bottom_left[0])/-2
    angle = math.degrees(math.atan(float(distance_left_midpoint_above_centroid)/float(distance_left_midpoint_from_center)))

    (x, y, w, h) = cv2.boundingRect(c)

    height_to_subtract = (abs(distance_left_midpoint_above_centroid) + 0.04 * h)
    width_to_subtract = int(w - math.cos(math.radians(angle)) * w + 0.09 * w)
    single_card = game[y:y+h, x:x+w]
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated_image = cv2.warpAffine(single_card, rotation_matrix, (w,h))
    rotated_cropped = rotated_image[height_to_subtract:h-height_to_subtract, width_to_subtract:w-width_to_subtract]

    cards.append(rotated_cropped)

for (i, c) in enumerate(cards):
    cardString = ''


    if c.shape[0] > c.shape[1]:
        c = cv2.transpose(c)

    graycard = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    _,threshcard = cv2.threshold(graycard, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('threshcard', threshcard)

# contours calculated

    cardContoursAll, hierarchy = cv2.findContours(threshcard, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cardContoursExternal, hierarchy = cv2.findContours(threshcard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# evaluate shape count using external contour count

    cardString += str(len(cardContoursExternal))

# evaluate texture by comparing total contour count to external count

    if len(cardContoursAll) == len(cardContoursExternal):
        cardString += 'sol'
    elif len(cardContoursAll) == (len(cardContoursExternal) * 2):
        cardString += 'emp'
    elif len(cardContoursAll) > (len(cardContoursExternal) * 2):
        cardString += 'str'

# evaluate shape by comparing to template

    diamondMatch = cv2.matchShapes(diamondContours[0], cardContoursExternal[0], 3, 0.0)
    ovalMatch = cv2.matchShapes(ovalContours[0], cardContoursExternal[0], 3, 0.0)
    squiggleMatch = cv2.matchShapes(squiggleContours[0], cardContoursExternal[0], 3, 0.0)

    if diamondMatch < ovalMatch and diamondMatch < squiggleMatch:
        cardString += 'Diamond'
    elif ovalMatch < squiggleMatch:
        cardString += 'Oval'
    else:
        cardString += 'Squiggle'

# evaluate color (WIP)

    pixelSample = c[cardContoursExternal[0][0][0][1], cardContoursExternal[0][0][0][0]]
    if pixelSample[0] > pixelSample[1] and pixelSample[0] > pixelSample[2]:
        cardString += 'Purple'
    elif pixelSample[1] > pixelSample[2]:
        cardString += 'Green'
    else:
        cardString += 'Red'

    cv2.imshow('current card', c)
    print cardString
    cv2.waitKey()

cv2.waitKey()
cv2.destroyAllWindows()
