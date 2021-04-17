import cv2
ds_factor=0.6
import numpy as np
import dlib
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor.dat")



def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right()-x
        h = rect.bottom()-y
        return (x,y,w,h)
def shape_to_np(shape,dtype="int"):
    coords = np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i] = (shape.part(i).x,shape.part(i).y)
    return coords

def smile(shape):
    left = shape[48]
    right = shape[54]
    mid = (shape[51] + shape[62] +shape[66] +shape[57])/4
    dist = np.abs(np.cross(right-left,left - mid)/np.linalg.norm(right-left))
    return dist

def faceLandmarkPoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    cv2.putText(image," No Mask Detected",(60,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(-1)
    
    def __del__(self):
        self.video.release()


    
    def get_frame(self):

        (mStart, mEnd) = (48,67)

        smile_const = 6
        #counter = 0 #when counter reaches 15 frames a selfie will be captured
        #selfie_no = 0


        success, image = self.video.read()
        image = cv2.flip(image,1)
        ogimage = image
        
        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        

        rects = detector(gray, 1)
        for i in range (0, len(rects)):
            (x,y,w,h) = rect_to_bb(rects[i])
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(image,"Face #{}".format(i+1), (x-20,y-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            shape = predictor(gray,rects[i])
            faceLandmarkPoints(image,shape)
            shape = shape_to_np(shape)
            mouth = shape[mStart:]
            for (x,y) in mouth:
                cv2.circle(image,(x,y),1,(255,255,255),-1)
                smile_param = smile(shape)
                cv2.putText(image,"SP: {:.2f}".format(smile_param),(300,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                if(smile_param > smile_const):
                    cv2.putText(image,"Smile Detected",(180,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    #counter += 1
                    ##if counter >=5:
                        #selfie_no +=1
                        #ret, frame = cam.read()
                    #image_path = r'C:/Users/Shubham/Desktop/project2/VideoStreamingFlask/smart_selfie.png'
                    #directory = r'C:/Users/Shubham/Desktop/project2/VideoStreamingFlask/static/smart_selfie.png'

                    #img = cv2.imread(image_path)
                    #os.chdir(directory) 
                    #path ='C:/Users/Shubham/Desktop/project2/VideoStreamingFlask/static'
                    
                    img_name = "smart_selfie.png"
                    #cv2.imwrite(os.path.join(path ,img_name),image)
                    cv2.imwrite(img_name,image)

                    original_frame_name = "original.png"
                    cv2.imwrite(original_frame_name,ogimage)

                    
                    #haarcas_output = "haarcas_selfie.png"
                    #cv2.imwrite(haarcas_output,haar_image)
                    #print("taken!")
                        #counter = 0
                #else:
                 
              # counter = 0
            
            break 
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
