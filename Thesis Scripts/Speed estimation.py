import cv2 as cv
import numpy as np
import csv
import concurrent.futures

def getLocation(bounding_box):
    xlt, ylt, xrb, yrb = bounding_box
    return (int((xlt + xrb) / 2.0), int((ylt + yrb) / 2.0))

# applying camera calibration and speed estimation
# arguments
# lst_disp_pixels : points in the frame
# frame_difference : difference among the frame captured if 60fps and 15 difference then it will be = 4
def calibrate_camera(lst_disp_pixels,frame_difference):
    real_displacement=[]
    for i in range(len(lst_disp_pixels)-1):
        displacement_pixels=abs(lst_disp_pixels[i+1]-lst_disp_pixels[i])
        real_displacement.append((displacement_pixels*57.35)/3700.78)
    speed_list=[]
    for j in range(len(real_displacement)):
        speed_list.append(real_displacement[j]*frame_difference)# replace frame difference here
        # if frame difference is 3 then multiply with 20
    avg_speedmps=sum(speed_list)/len(speed_list)
    avg_speedkmph=avg_speedmps*3.6
    return format(avg_speedkmph,'.2f')

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    # print(file_name)
    try:
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            # print(len(list_of_elem))
            csv_writer = csv.writer(write_obj, escapechar='/', quoting=csv.QUOTE_NONE)
            # Add contents of list as last row in the csv file
            # print("content writing")
            csv_writer.writerow(list_of_elem)
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")

def detect_Objects(old_frame):
    yolomodel = {"config_path":"yolo-obj.cfg",
                  "model_weights_path":"yolov4-obj_final.weights",
                  "coco_names":"obj.names",
                  "confidence_threshold": 0.5,
                  "threshold":0.3
                 }

    net = cv.dnn.readNetFromDarknet(yolomodel["config_path"], yolomodel["model_weights_path"])
    labels = open(yolomodel["coco_names"]).read().strip().split("\n")

    np.random.seed(12345)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    print(layer_names)

    bbox_colors = np.random.randint(0, 255, size=(len(labels), 3))




    W,H=None,None
    if W is None or H is None: (H, W) = old_frame.shape[:2]
    blob = cv.dnn.blobFromImage(old_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)  # detect objects using object detection model

    detections_bbox = []  # bounding box for detections
    boxes, confidences, classIDs = [], [], []
    for out in detections_layer:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > yolomodel['confidence_threshold']:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, yolomodel["confidence_threshold"], yolomodel["threshold"])
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            detections_bbox.append((x, y, x + w, y + h))
            clr = [int(c) for c in bbox_colors[classIDs[i]]]
            cv.rectangle(old_frame, (x, y), (x + w, y + h), clr, 2)
            cv.putText(old_frame, "{}".format(labels[classIDs[i]]),
                       (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

    new_object_locations = np.zeros((len(detections_bbox), 1, 2), dtype="float32")  # current object locations

    for (i, detection) in enumerate(detections_bbox): new_object_locations[i] = getLocation(detection)
    return new_object_locations



# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                           10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
video_src = 'multi-object-tracker-master/video_data/DJI_0051.MOV'#0
cap = cv.VideoCapture(video_src)
# Create a mask image for drawing purposes
ret, old_frame = cap.read()
mask = np.zeros_like(old_frame)
old_gray = cv.cvtColor(old_frame,
                       cv.COLOR_BGR2GRAY)

new_object_locations=detect_Objects(old_frame)

frame_count=1

while (1):

    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame,
                              cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           new_object_locations, None,
                                           **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = new_object_locations[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d),
                        color[i].tolist(), 2)
        #resize = cv.resize(frame, (640, 480), interpolation=cv.INTER_LINEAR)
        frame = cv.circle(frame, (a, b), 5,
                           color[i].tolist(), -1)
        #cv.putText(frame, "{}".format(str(calibrate_camera([b,d],60))+","+str(i)),
        #           (int(a), int(b) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, color[i].tolist(), 4)
        cv.putText(frame, "{}".format(str(calibrate_camera([b, d], 60)) + "," + str(i)),
                    (int(a), int(b) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 4)
        row_contents = [str(frame_count), str(i), str(a)+","+str(b), str(c)+","+str(d)]
        # Append a list as new line to an old csv file
        append_list_as_row('speedEstOpticalflow.csv', row_contents)

    img = cv.add(frame, mask)
    frame_count = frame_count + 1
    cv.imshow('frame', img)

    k = cv.waitKey(25)
    print("frame count = {}".format(frame_count))
    if k == 27:
        break

    # if frame_count%30==0:
    #     break

    # Updating Previous frame and points
    old_gray = frame_gray.copy()
    new_object_locations = good_new.reshape(-1, 1, 2)
    if frame_count%50==0:
        cv.imwrite("frame%d.jpg" % frame_count, frame)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(detect_Objects, frame)
            new_object_locations = future.result()
            #print(return_value)
        #new_object_locations=detect_Objects(frame)

cv.destroyAllWindows()
cap.release()