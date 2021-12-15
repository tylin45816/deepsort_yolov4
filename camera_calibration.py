import cv2
import numpy as np
from numpy.lib.function_base import append
from torch import dist
from numpy.lib.type_check import imag


calculateLine1 = [(310,417),(847,430)]  #[(from(x,y), to(x,y)]
calculateLine2 = [(310,417),(140,747)]  #[(from(x,y), to(x,y)]
calculateLine3 = [(140,740),(1050,740)]  #[(from(x,y), to(x,y)]
calculateLine4 = [(1050,740),(847,430)]  #[(from(x,y), to(x,y))
# url = "http://control.bote.gov.taipei/view_camera.html?ccd_id=205"
# # url = "https://thbcctv06.thb.gov.tw/T15-1K+000"
def draw_CalculateLine(frame):
    offset = 20
    cv2.line(frame, (calculateLine1[0][0]+offset,calculateLine1[0][1]+offset), (calculateLine1[1][0]-offset,calculateLine1[1][1]+offset), (0, 255, 0), 20)
    cv2.line(frame, (calculateLine2[0][0]+offset,calculateLine2[0][1]+offset), (calculateLine2[1][0]+30,calculateLine2[1][1]-offset), (0, 255, 0), 20)
    cv2.line(frame, (calculateLine3[0][0],calculateLine3[0][1]-offset), (calculateLine3[1][0],calculateLine3[1][1]-offset), (0, 255, 0), 20)
    cv2.line(frame, (calculateLine4[0][0],calculateLine4[0][1]+offset), (calculateLine4[1][0],calculateLine4[1][1]+offset), (0, 255, 0), 20)

    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]), (calculateLine1[1][0],calculateLine1[1][1]), (0, 0, 255), 20)
    cv2.line(frame, (calculateLine2[0][0],calculateLine2[0][1]), (calculateLine2[1][0],calculateLine2[1][1]), (0, 0, 255), 20)
    cv2.line(frame, (calculateLine3[0][0],calculateLine3[0][1]), (calculateLine3[1][0],calculateLine3[1][1]), (0, 0, 255), 20)
    cv2.line(frame, (calculateLine4[0][0],calculateLine4[0][1]), (calculateLine4[1][0],calculateLine4[1][1]), (0, 0, 255), 20)
    return frame

def cal_point_dis(p1,p2):
    dis = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dis

def  coordinate_transform(matrix,ori_x,ori_y):
    matrix = np.array(matrix)
    ori_coord = np.array(([ori_x],[ori_y],[1]))
    result = np.dot(matrix,ori_coord)

    x,y = result[0]/result[2] , result[1]/result[2]
    return x,y

cap = cv2.VideoCapture('55.mp4')
y1 = []
y2 = []
y3 = []
x1 = []
x2 = []
h1 = []
h2 = []
v1 = []
v2 = []
while (True):
    interval = 1
    # logo = imutils.url_to_image(url)
    ret, normal_frame = cap.read()
    height, width = normal_frame.shape[:2]
    print(height)
    print(width)
    # normal_frame = np.uint8(np.clip((normal_frame),0,255))
    
    # fImg = normal_frame.astype(np.float32)
    # fImg = fImg / 255.0
    # # 顏色空間轉換 BGR -> HLS
    # hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    # hlsCopy = np.copy(hlsImg)
    # lightness = 20 # lightness 調整為  "1 +/- 幾 %"
    # saturation = 50 # saturation 調整為 "1 +/- 幾 %"
    # # 亮度調整
    # hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    # hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
    # # 飽和度調整
    # hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    # hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
    # # 顏色空間反轉換 HLS -> BGR 
    # result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    # normal_frame = ((result_img * 255).astype(np.uint8))

    # for i in range(10):
        # y1.append((270-6*i,250+5*i))
        # y2.append((366-4*i,252+5*i))
        # y3.append((462-2*i,254+5*i))
        # normal_frame = cv2.circle(normal_frame,y1[i],2,(0,0,255),cv2.FILLED)
        # normal_frame = cv2.circle(normal_frame,y2[i],2,(0,0,255),cv2.FILLED)
        # normal_frame = cv2.circle(normal_frame,y3[i],2,(0,0,255),cv2.FILLED)
        # normal_frame = cv2.line(normal_frame,y1[i],y3[i],(255,0,0),1,cv2.LINE_AA)

    # for i in range(width-1):
    #     if i%interval==0:
    #         v1.append((i,0))
    #         v2.append((i,height))
    # for i in range(int(width/interval  )):
    #     normal_frame = cv2.line(normal_frame,v1[i],v2[i],(255,0,0),1,cv2.LINE_AA)
    # for j in range(height-1):
    #     if j%interval==0:
    #         h1.append((0,j))
    #         h2.append((width,j))
    # for j in range(int(height/interval)):
    #     normal_frame = cv2.line(normal_frame,h1[j],h2[j],(255,0,0),1,cv2.LINE_AA)

    # normal_frame = cv2.circle(normal_frame,(379,404),2,(0,0,255),cv2.FILLED)
    # normal_frame = cv2.circle(normal_frame,(774,414),2,(0,0,255),cv2.FILLED)
    # normal_frame = cv2.circle(normal_frame,(1026,731),2,(0,0,255),cv2.FILLED)
    # normal_frame = cv2.circle(normal_frame,(195,735),2,(0,0,255),cv2.FILLED)

    normal_frame = cv2.circle(normal_frame,(108,340),2,(0,0,255),cv2.FILLED)
    normal_frame = cv2.circle(normal_frame,(205,342),2,(0,0,255),cv2.FILLED)
    normal_frame = cv2.circle(normal_frame,(302,344),2,(0,0,255),cv2.FILLED)
    normal_frame = cv2.circle(normal_frame,(400,346),2,(0,0,255),cv2.FILLED)

    # normal_frame = cv2.circle(normal_frame,(600,560),2,(0,0,255),cv2.FILLED)

    # cv2.fillPoly(normal_frame,np.array([[(379,404),(582,434),(600,560),(51,568)]]),(0,255,0)) 
    # cv2.fillPoly(normal_frame,np.array([[(774,414),(1079,545),(600,560),(582,434)]]),(255,255,0)) 
    # cv2.fillPoly(normal_frame,np.array([[(1026,731),(637,742),(600,560),(1079,545)]]),(0,255,255)) 
    # cv2.fillPoly(normal_frame,np.array([[(195,735),(51,568),(600,560),(637,742)]]),(255,0,255)) 

    # points1 = np.float32([(457,89),(628,79),(91,410),(893,348)])
    points1 = np.float32([[(173,262)],[(369,269)],[(108,340)],[(400,346)]])
    # points2 = np.float32([[400,100],[400+int(width/4),100],[400,100+int(height/2)],[400+int(width/4),100+int(height/2)]])
    points2 = np.float32([[0,0],[int(width),0],[0,height],[int(width),height]])

    M = cv2.getPerspectiveTransform(points1,points2)
    processed = cv2.warpPerspective(normal_frame,M,(int(width),int(height)),cv2.INTER_LINEAR)
    cv2.imshow('processed',processed)
    cv2.imshow('normal_data',normal_frame)

    # ''' camera calibration '''
    # # points1 = np.float32([[230,250],[400,250],[40,410],[350,400]])
    # # points1 = np.float32([[280,250],[465,250],[90,410],[400,400]])
    # points1 = np.float32([[281,250],[467,253],[226,310],[443,310]])


    # # width = cal_point_dis([270,250],[790,240])
    # # length = cal_point_dis([400,130],[145,360])

    # points2 = np.float32([[0,0],[width/2,0],[0,height/2],[width/2,height/2]])
    # trans_matrix = cv2.getPerspectiveTransform(points1,points2)
    # camera_calibration = cv2.warpPerspective(normal_frame,trans_matrix,(int(width),int(height)),cv2.INTER_LINEAR)

    # x,y = coordinate_transform(trans_matrix,333,297)
    # print('result',x,y)

    # '''draw calculate line'''
    # normal_frame = draw_CalculateLine(normal_frame)



    # normal_frame = cv2.line(normal_frame,(130,376),(404,395),(255,0,0),3,cv2.LINE_AA)
    # normal_frame = cv2.line(normal_frame,(25,465),(358,495),(255,0,0),3,cv2.LINE_AA)
    # normal_frame = cv2.line(normal_frame,(270,250),(465,250),(255,0,0),3,cv2.LINE_AA)
    # normal_frame = cv2.line(normal_frame,(270,250),(190,320),(255,0,0),3,cv2.LINE_AA)
    # normal_frame = cv2.line(normal_frame,(190,320),(443,326),(255,0,0),3,cv2.LINE_AA)
    # normal_frame = cv2.line(normal_frame,(465,250),(443,326),(255,0,0),3,cv2.LINE_AA)
    # normal_frame = cv2.line(normal_frame,(90,410),(400,420),(255,0,0),3,cv2.LINE_AA)

    # cv2.imshow('camera_calibration',camera_calibration)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

