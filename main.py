import cv2
import torch
import numpy as np
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

fromaddr="tdinesh986@gmail.com" 


toaddr= "jananishankar0706@gmail.com"


def mail(text):
    print(text)
    msg=MIMEMultipart()
    msg['From']=fromaddr
    msg['To']=toaddr
    msg['Subject']="NOT_SAFETY"
    body=text
    msg.attach(MIMEText(body,'plain'))
    filename="output/img.jpg"
    attachment=open("output/img.jpg","rb")
    p=MIMEBase('application','octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition',"attachment; filename=%s"%filename)
    msg.attach(p)
    s=smtplib.SMTP('smtp.gmail.com',587)
    s.starttls()
    s.login(fromaddr,"okntacnixyiqmldc") 
    text=msg.as_string()
    s.sendmail(fromaddr,toaddr,text)
    s.quit()


def detect_objects_live(weights_path='best.pt', conf_threshold=0.2):
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Open video capture device (webcam)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Get bounding boxes, confidence scores, and class labels
        boxes = results.xyxy[0]  # Bounding boxes in (x1, y1, x2, y2) format
        confidences = boxes[:, 4]  # Confidence scores
        class_labels = boxes[:, 5]  # Class labels

        # Filter detections based on confidence threshold
        detections_above_threshold = boxes[confidences > conf_threshold]

        # Draw bounding boxes for detections above threshold
        for detection in detections_above_threshold:
            label = int(detection[5])
            score = float(detection[4])
            bbox = detection[:4].cpu().numpy().astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[label]}: {score:.2f}',
                        (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # print(model.names[label])
            klass=model.names[label]

            # NO-Safety Vest,NO-Mask,NO-Hardhat
            if klass == "NO-Hardhat":
                cv2.imwrite('output/img.jpg', frame)
                print("Without Hardhat")
                mail("NOT_SAFETY")
            elif klass == "NO-Safety Vest":
                cv2.imwrite('output/img.jpg', frame)
                print("Without Safety Vest")
                mail("NOT_SAFETY")
            elif klass == "NO-Mask":
                cv2.imwrite('output/img.jpg', frame)
                print("Without Mask")
                mail("NOT_SAFETY")

        # Display the frame
        cv2.imshow('Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_objects_live()
