import cv2
import numpy as np

diamond = cv2.imread('shapeTemplates/diamond.jpg')
oval = cv2.imread('shapeTemplates/oval.jpg')
squiggle = cv2.imread('shapeTemplates/squiggle.jpg')

ret, diamondThresh = cv2.threshold(cv2.imread('shapeTemplates/diamond.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
ret, ovalThresh = cv2.threshold(cv2.imread('shapeTemplates/oval.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
ret, squiggleThresh = cv2.threshold(cv2.imread('shapeTemplates/squiggle.jpg', 0), 127, 255, cv2.THRESH_BINARY_INV)
diamondContours, hierarchy = cv2.findContours(diamondThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
ovalContours, hierarchy = cv2.findContours(ovalThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
squiggleContours, hierarchy = cv2.findContours(squiggleThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


ovalTemplate = cv2.imread('shapeTemplates/oval.jpg')
squiggleTemplate = cv2.imread('shapeTemplates/squiggle.jpg')


game1 = cv2.imread('setgame2.jpg')

game1gray = cv2.cvtColor(game1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(game1gray, 150, 255, 0)

kernel = np.ones((4,4), np.uint8)

eroded = cv2.erode(thresh, kernel, iterations = 1)

contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

print "contours found:", len(contours)

cv2.drawContours(game1, contours, -1, (0, 255, 0), 3)

cards = []

for (i, c) in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(c)
    cropped_contour = game1[y+10:y+(h-10), x+10:x+(w-10)]
    cards.append(cropped_contour)

for (i, c) in enumerate(cards):
    cardString = ''

# preprocessing

    if c.shape[0] > c.shape[1]:
        c = cv2.transpose(c)
    graycard = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    ret, threshcard = cv2.threshold(graycard, 160,255, cv2.THRESH_BINARY_INV)

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

    cv2.imshow('current card', c)
    print cardString
    cv2.waitKey()

cv2.waitKey()
cv2.destroyAllWindows()
