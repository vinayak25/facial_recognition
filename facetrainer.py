import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.createLBPHFaceRecognizer();
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path='dataSet'

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        #Loading the image and converting it to grey scale
        pilImage=Image.open(imagePath).convert('L');
        #Now we are converting the PIL image into numoy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split('.')[1]) #counts from backward
        #extract the face from trainig image sample
        faces=detector.detectMultiScale(imageNp)
        #if a face is there the append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
            cv2.imshow("training",imageNp)
            cv2.waitKey(10)
    return faceSamples,Ids

faces,Ids=getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
