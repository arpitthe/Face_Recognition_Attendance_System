import cv2

face = r"C:\Users\Acer\Downloads\haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)
ret , frame = cap.read()

while ret:
    ret , frame = cap.read()
    print(ret)
    print(frame)
    if not ret:
        break
    model = cv2.CascadeClassifier(face)
    facepoints = model.detectMultiScale(frame)
    for (x,y,w,h) in facepoints:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow('face',frame)
    if cv2.waitKey(10)==ord('q'):
        break