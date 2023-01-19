import cv2
import numpy as np
import utlis

path = "1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
answers = [1, 2, 0, 1, 4]

# ÖN IŞLEMLER
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgFinal = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# KONTUR BULMA
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# DÖRTKEN BULMA
rectCon = utlis.rectContour(contours)
biggestContour = utlis.getCornerPoints(rectCon[0])
gradePoints = utlis.getCornerPoints(rectCon[1])
#print(biggestContour)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utlis.reorder(biggestContour)
    gradePoints = utlis.reorder(gradePoints)

    point1 = np.float32(biggestContour)
    point2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(point1, point2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    pointGrade1 = np.float32(gradePoints)
    pointGrade2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixGrade = cv2.getPerspectiveTransform(pointGrade1, pointGrade2)
    imgGradeDisplay = cv2.warpPerspective(img, matrixGrade, (325, 150))
    #cv2.imshow("grade", imgGradeDisplay)

    # TRESHOLD UYGULAMA
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    # HER KUTUNUN PIXEL DEĞERLERİNİ ALMA
    boxes = utlis.splitBoxes(imgThresh)
    myPixelVal = np.zeros((questions, choices))
    countColumn = 0
    countRow = 0
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countRow][countColumn] = totalPixels
        countColumn += 1
        if (countColumn == choices):
            countRow += 1
            countColumn = 0
    #print(myPixelVal)

    # ŞIKLARIN İNDEKS DEĞERLERİNİ BULMA
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
        #print(myIndex)

    # NOTLANDIRMA
    grading = []
    for x in range(0, questions):
        if answers[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    #print(grading)
    score = (sum(grading) / questions) * 100
    #print(score)


    # CEVAPLARI GÖSTERME
    imgResult = imgWarpColored.copy()
    imgResult = utlis.showAnswers(imgResult, myIndex, grading, answers, questions, choices)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, answers, questions, choices)
    invMatrix = cv2.getPerspectiveTransform(point2, point1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade, str(int(score))+"%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    invMatrixGrade = cv2.getPerspectiveTransform(pointGrade2, pointGrade1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixGrade, (widthImg, heightImg))
    
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

# IMAGE GÖSTERME
imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
              [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
imageStacked = utlis.stackImages(imageArray, 0.3)

cv2.imshow("Final Image", imgFinal)
cv2.imshow("Stacked Images", imageStacked)
cv2.waitKey(0)