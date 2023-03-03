import cv2
import numpy as np
import matplotlib.pyplot as plt



min_width_rect = 80
min_height_rect =80

# initialize algorithm
algo = cv2.createBackgroundSubtractorMOG2(detectShadows=False)



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    canny = cv2.Canny(blur, 5, 150)
    cv2.imshow("gray", canny)
    return canny

# cv2.imshow("gray", gray)
# cv2.imshow("gray", blur)
# cv2.imshow("gray", canny)



def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(198,804 ), (350, 40), (580, 30), (700, 650)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image
    


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return line_image


# image = cv2.imread("RingRoad.jpg")
# lane_image = np.copy(image)
# canny = canny(lane_image)
# cropped_image = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
#                         np.array([]), minLineLength=40, maxLineGap=5)
# line_image = display_lines(lane_image, lines)
# combo_image = cv2.addWeighted(lane_image, 1, line_image, 1, 1)

# for coordinates  in lines:
#     print(coordinates)

# cv2.imshow("result", combo_image)
# cv2.waitKey(0)
#plt.imshow(combo_image)
#plt.show()
def start():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        cropped_image = region_of_interest(frame)
        canny_image = canny(cropped_image)
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=50)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)
        
        
                    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(frame, (3, 3), 5)
        
        #line coordinates for left line and right line

        # apply on all frame
        image_sub = algo.apply(cropped_image)
        dilat = cv2.dilate(image_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        countour_shape,h = cv2.findContours(
            dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("combo_image",image_sub)

        for (i, c) in enumerate(countour_shape):
            (x, y, w, h) = cv2.boundingRect(c)
            valid_vech = (w >= min_width_rect) and (h >= min_height_rect)
            if not valid_vech:
                continue
            
            rectangle_draw = cv2.rectangle(combo_image, (x,y), (x+h, y+h), (0,255,0), 2)
            #roi= (x, y, x+h, y+h)
            
            line1=cv2.line(combo_image, (x,y),(x,y+h), (255,0,0))
            x5=560
            y5=30
            x2=665
            y2=650

            # x3=1012
            # y3=560
            # x4=978
            # y4=301


            draw_line1=cv2.line(combo_image,(x5,y5),(x2,y2),(255,0,0,),1)
            # #draw_line2=cv2.line(combo_image,(x3,y3),(x4,y4),(0,255,0),1)
            # if(x<x5):
            #     print("vehicle crossed lane")
            #     # vehicle_image = combo_image[x:w, y:h]
            #     # cv2.imwrite('.jpg',vehicle_image)
            #     # #vehicle_image.save("lanecrossed.jpg")
            #     # cv2.imshow("cropped_image",vehicle_image)
            # else:
            #     print("lane is not crossed")
                

        
            
            
            
            
            
    
            
            
            
            
                
            
            
            
            
            
            
            
        #cv2.imshow('dilatada', dilatada)
        #cv2.imshow('original video', frame)
        
        
        #plt.imshow(traf.png)
        cv2.imshow("result", combo_image)
        
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



        



    # min_width_rect = 80
    # min_height_rect =80

    # # initialize algorithm
    # algo = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=500)


    # while True:
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     blur = cv2.GaussianBlur(frame, (3, 3), 5)

    #     # apply on all frame

    #     image_sub = algo.apply(blur)
    #     dilat = cv2.dilate(image_sub, np.ones((5, 5)))
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    #     dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    #     countour_shape,h = cv2.findContours(
    #         dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #     for (i, c) in enumerate(countour_shape):
    #         (x, y, w, h) = cv2.boundingRect(c)
    #         valid_vech = (w >= min_width_rect) and (h >= min_height_rect)
    #         if not valid_vech:
    #             continue
            
    #         cv2.rectangle(frame, (x,y), (x+h, y+h), (0,255,0), 2)
            

    #     #cv2.imshow('dilatada', dilatada)
    #     cv2.imshow('original video', frame1)

    #     if cv2.waitKey(1) == 13:
    #         break

    # cv2.destroyAllWindows()
    # cap.release()