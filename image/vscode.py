import cv2
image = cv2.imread('grp_pic.jpg')
classifier = cv2.CascadeClassifier('haarcascade_frontalface.xml')
gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
bboxes = classifier.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=10)
print(bboxes)
print("face detected")

for x,y,w,h in bboxes:
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    
resized = cv2.resize(image,(int(image.shape[1]/2), int(image.shape[0]/2)))
    
cv2.imshow("picture", resized)
cv2.waitKey(0)
cv2.destroyAllWindows
