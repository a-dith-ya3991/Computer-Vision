import cv2
import numpy as np

# Define range of orange color in HSV
lower_orange = np.array([0, 150, 100])
upper_orange = np.array([40, 255, 255])

# Read video file
cap = cv2.VideoCapture(r"video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'AVC1')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(r'output.mp4', fourcc, 20.0, (width, height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

    
        blur = cv2.GaussianBlur(mask, (9, 9), 0)
        circ = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=15, param2=33, minRadius=0, maxRadius=120)

        if circ is not None:
            circ=circ.reshape((circ.shape[0]*circ.shape[1],3))
            circ=np.round(circ).astype('int')
            l=list(map(lambda x: list(x),circ))
            ls=sorted(l,key=lambda x: x[-1],reverse=True)
            
            (x,y,r)=ls[0]
            if r>50:
                min_row=x-r
                min_col=y-r
                max_row=x+r
                max_col=y+r
                cv2.rectangle(frame,(min_row,min_col),(max_row,max_col),(0,255,0),3)
                t_r=min_row-7
                t_c=min_col
                
                cv2.putText(frame ,"circle",(t_r,t_c),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        # Display the resulting frame
        cv2.imshow('Orange Circle Detection', frame)
        out.write(frame)
        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
out.release()
