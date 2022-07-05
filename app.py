from flask import Flask, render_template, Response
import cv2
app=Flask(__name__)
camera = cv2.VideoCapture("http://192.168.8.101:8080/video")
#camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            thres = 0.45  # Threshold to detect object
            nms_threshold = 0.2

            classNames = []
            classFile = 'coco.names'
            with open(classFile, 'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')
            #print(classNames)

            configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
            weightsPath = 'frozen_inference_graph.pb'

            net = cv2.dnn_DetectionModel(weightsPath, configPath)
            net.setInputSize(320, 320)
            net.setInputScale(1.0 / 127.5)
            net.setInputMean((127.5, 127.5, 127.5))
            net.setInputSwapRB(True)

            # success, img = camera.read()
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)
            # print(classIds, bbox)

            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                #cv2.imshow("Output", img)
                #cv2.waitKey(1)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
