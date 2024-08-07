import playsound
from ultralytics import YOLO
import cvzone,time
import cv2
import math
import smtplib 
import threading

mail = False
flag = True


cap = cv2.VideoCapture('fire4.mp4')
# cap = cv2.VideoCapture(0)   CHANGE TO TAKE INPUT FROM CAMERA
model = YOLO('fire.pt')

def play_alarm_sound_function(): 
    playsound.playsound('Alarm Sound.mp3',True) 
     
    
    print("Fire alarm end")

def send_mail_function():  
    recipientmail = "ENTER TO MAIL ID"  
    recipientmail = recipientmail.lower()  

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login("FROM ID", 'PASSWORD (ONETIME KEY )')  
        server.sendmail('FROM ID', recipientmail, "Warning fire accident has been reported")  
        print("Alert mail sent successfully to {}".format(recipientmail))  
        server.close()  

    except Exception as e:
        print(e)  

# Reading the classes
classnames = ['fire']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

   
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 90:
                mail = True
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)
            while mail == True and flag == True:
                print("Fire alarm initiated")
    
                threading.Thread(target=play_alarm_sound_function).start() 
               
                print("Mail send initiated")
                threading.Thread(target=send_mail_function).start()  
                mail = False
                flag=False

            

    cv2.imshow('frame', frame)
    cv2.waitKey(1)