import cv2 as cv
from ultralytics import YOLO

def procesVideo():
    detection_classes = []
    path = r"C:\Users\LENOVO\OneDrive\Desktop\YOLO\Tracking\281621.mp4"
    vs = cv.VideoCapture(path)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 960, 540)
    model=YOLO(r"C:\Users\LENOVO\OneDrive\Desktop\YOLO\Tracking\yolov8n.pt")

    output_path = r"C:\Users\LENOVO\OneDrive\Desktop\YOLO\Tracking\output.mp4"
    fps = vs.get(cv.CAP_PROP_FPS) or 30
    width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        results=model.predict(frame,stream=False)
        print(results[0].names)
        detection_classes = results[0].names 

        for result in results:
            for data in result.boxes.data.tolist():
                #print(data)
                id=data[5]
                drawbox(data,frame,detection_classes[id])
                
                print("detected class: ",detection_classes[id])
        out.write(frame) 
        cv.imshow('image', frame)
        if cv.waitKey(24) & 0xFF == ord('q'):
            break

    vs.release()
    out.release()
    cv.destroyAllWindows()
    print("Saved to:", output_path)

def drawbox(data,image,name):
    x1,y1,x2,y2,score,class_id=data
    p1=(int(x1),int(y1))
    p2=(int(x2),int(y2))
    cv.rectangle(image,p1,p2,(0,255,0),3)
    cv.putText(image,f"{name} {score:.2f}",(int(x1),int(y1)-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return image
    

procesVideo()
