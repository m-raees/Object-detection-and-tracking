import cv2
import os

dirname="DJI_0027_FRAMES"

if not os.path.exists(dirname):
  os.mkdir(dirname)

vidcap = cv2.VideoCapture('DJI_0027.MOV')
success,image = vidcap.read()

count = 0
while success:
    if count%15==0:
        cv2.imwrite(os.path.join(dirname, "27_frame%d.png" % count), img=image)
        print('Read a new frame:  '+str(count)+"    "+ str(success))
    success, image = vidcap.read()
    count += 1
    if count==5000:
        break