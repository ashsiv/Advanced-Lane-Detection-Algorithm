import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import time

src=np.float32([[250,700],[1150,700],[580,450],[720,450]])
dst=np.float32([[250,700],[1150,700],[250,0],[1150,0]])
for file in glob.glob('test_images/*.jpg'):
    img = mpimg.imread(file)
    
    
    
   
M=cv2.getPerspectiveTransform(src,dst)


x=250
y=700

def remapsrc(left_fitx,ploty):
    x1=[]
    y1=[]
    for i in range(len(left_fitx)):
        x=left_fitx[i]
        y=ploty[i]
        x1.append(round(((M[0][0]*x)+(M[0][1]*y)+(M[0][2]))/((M[2][0]*x)+(M[2][1]*y)+M[2][2]),0))
        y1.append(round(((M[1][0]*x)+(M[1][1]*y)+(M[1][2]))/((M[2][0]*x)+(M[2][1]*y)+M[2][2]),0))
        print(x1)
        print(y1)
    src=np.transpose(np.vstack((x1,y1)))

