from cv2 import VideoCapture, imwrite, imshow, waitKey, destroyWindow, CAP_PROP_POS_FRAMES
import time
import requests

cam_port = 0
cam = VideoCapture(cam_port)

url = "http://172.26.61.181:3237/predict"

session = requests.Session()

while (cam.isOpened()):
    cf = cam.get(CAP_PROP_POS_FRAMES) - 1
    cam.set(CAP_PROP_POS_FRAMES, cf+50)
    result, image = cam.read()
    if result:  
        # showing result, it take frame name and image  
        # output 
        # imshow("cam_out", image) 
    
        # saving image in local storage 
        imwrite("cam_out.png", image) 

        files = {'image': open('cam_out.png', 'rb')}

        ans = session.post(url, files=files)

        print("Image captured", ans.json()['prediction'])
    
        # If keyboard interrupt occurs, destroy image  
        # window 
        time.sleep(15)

        # destroyWindow("cam_out") 

    else:
        print("No image detected. Please! try again") 
        break

session.close()
cam.release()
# cv2.destroyAllWindows()