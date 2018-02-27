import cv2

face = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('../cascades/haarcascade_eye.xml')

if face.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

if eyes.empty():
    raise IOError('Unable to load the eyes cascade classifier xml file')

img = cv2.imread('../cache/extract/face.zip/gt_db/s01/01.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detected = face.detectMultiScale(gray, 1.3, 5)
eyes_detected = eyes.detectMultiScale(gray, 1.3, 5)
print(eyes_detected)

for (x, y, w, h) in face_detected:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

for (x, y, w, h) in eyes_detected:
    cv2.circle(img, (x + int(h/2), y + int(w/2)), int(w/2), (255, 0, 0), 3)

cv2.imshow('face', img)
cv2.waitKey()
cv2.destroyAllWindows()
