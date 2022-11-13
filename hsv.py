# This script uses OpenCV to generate the upper and lower HSV ranges
    # of any pixel in an RGB image (tolerances could be modified in code).

import cv2
import numpy as np
import sys
import tkinter as tk
from   tkinter import filedialog

image_hsv = None
pixel     = (0,0 , 0) #RANDOM DEFAULT VALUE

ftypes = [
    ("JPG", "*.jpg;*.JPG;*.JPEG") ,
    ("PNG", "*.png;*.PNG")        ,
    ("GIF", "*.gif;*.GIF")        ,
    ("All files", "*.*")
]

def check_bound(value, toler, ranges, upper_or_lower):
    if ranges == 0:
        # set the bound for hue
        bound = 180
    elif ranges == 1:
        # set the bound for saturation and value
        bound = 255


    if(value + toler > bound):
        value = bound

    elif (value - toler < 0):
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + toler
        else:
            value = value - toler
    return value

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        # h,s,v ranges.
            # tolerance could be adjusted

        # set upper_or_lower = 1 for uppers
        # and upper_or_lower = 0 for lowers
        h_upper = check_bound(pixel[0], 10, 0, 1)
        h_lower = check_bound(pixel[0], 10, 0, 0)
                                            # range = 0 for hue and
                                            # range = 1 for saturation and brightness
        s_upper = check_bound(pixel[1], 10, 1, 1)
        s_lower = check_bound(pixel[1], 10, 1, 0)
        v_upper = check_bound(pixel[2], 40, 1, 1)
        v_lower = check_bound(pixel[2], 40, 1, 0)

        uppers =  np.array([h_upper, s_upper, v_upper])
        lowers =  np.array([h_lower, s_lower, v_lower])
        print(lowers, uppers)

        #a monochrome mask for getting a better vision over the colors
        image_mask = cv2.inRange( image_hsv, lowers, uppers )
        cv2.imshow( "Mask", image_mask )

def main():

    global image_hsv, pixel

    root = tk.Tk()  #open dialog for reading the image file
    root.withdraw() #hide the tkinter gui
    file_path = filedialog.askopenfilename(filetypes = ftypes)
    root.update()
    image_src = cv2.imread(file_path)
    cv2.imshow("BGR",image_src)

    image_hsv = cv2.cvtColor( image_src, cv2.COLOR_BGR2HSV )
    cv2.imshow("HSV",image_hsv)

    #callback function
    cv2.setMouseCallback("HSV", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
