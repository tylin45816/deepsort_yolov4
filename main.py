import argparse
from math import inf
import utils
from utils.sort.iou_matching import iou
from numpy.core.fromnumeric import shape
import torch
import cv2
import os
import time
import numpy as np
import random
import copy
from utils.utils import get_config, do_detect, plot_boxes,plot_boxes_cv2,load_class_names,bbox_iou
from utils.draw import draw_boxes
from utils import build_detector, build_deepsort
import my_vehicle

class VideoTracker(object):
    def __init__(self,cfg,args):
        self.args=args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            raise UserWarning("Running in cpu mode!")
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)
        self.vdo = cv2.VideoCapture()
        self.detector=build_detector(cfg,use_cuda=use_cuda)
        self.deepsort = build_deepsort(cfg, use_cuda=use_cuda)

    def sat(self,img):
        fImg = img.astype(np.float32)
        fImg = fImg / 255.0
        # 顏色空間轉換 BGR -> HLS
        hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
        hlsCopy = np.copy(hlsImg)
        lightness = 20 # lightness 調整為  "1 +/- 幾 %"
        saturation = 50 # saturation 調整為 "1 +/- 幾 %"
        # 亮度調整
        hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
        # 飽和度調整
        hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
        # 顏色空間反轉換 HLS -> BGR 
        result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        normal_frame = ((result_img * 255).astype(np.uint8))
        return normal_frame
    def _xywh_to_xyxy(self, bbox_xywh,width,height):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), height - 1)
        return x1, y1, x2, y2

    def cal_dist(delf,x1,y1,x2,y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def intersect(self,A,B,C,D):
        return VideoTracker.ccw(A,C,D) != VideoTracker.ccw(B,C,D) and VideoTracker.ccw(A,B,C) != VideoTracker.ccw(A,B,D)

    def __enter__(self):
        assert os.path.isfile(self.args.video_path), "Error: path error"
        self.vdo.open(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # if self.args.save_path:
        #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #     self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
    
    def bbox_transform(self,boxes,im):
        ''''transform the type of the bbox from yolov4_pytorch'''
        output = []
        wh = np.flip(im.shape[0:2])
        for bbox in boxes:
            cxcy = tuple((np.array(bbox[0:2]) * wh).astype(np.int64))
            bbx_wh = tuple((np.array(bbox[2:4]) * wh).astype(np.int64))
            cx, cy = cxcy
            w, h = bbx_wh
            x1 = int(abs(cx-w/2))
            y1 = int(abs(cy-h/2)*0.97)
            x2 = int(cx+w/2)
            y2 = int((cy+h/2)*1.03)
            classes = int(bbox[6])
            if x1 < 0:
                continue
            output.append([x1,y1,x2,y2,classes])
        return output

    def iou_bbox(self,bbox1,bbox2):
        iou_tl = [np.maximum(bbox1[0],bbox2[0]),np.maximum(bbox1[1],bbox2[1])]
        iou_br = [np.minimum(bbox1[2],bbox2[2]),np.minimum(bbox1[3],bbox2[3])]
        iou_w = np.maximum((iou_br[0] - iou_tl[0]) , 0)
        iou_h = np.maximum((iou_br[1] - iou_tl[1]) , 0)
        area_intersection = iou_w * iou_h
        area_bbox1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        area_bbox2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        area_union = (area_bbox1 + area_bbox2 - area_intersection)
        iou = np.maximum((area_intersection / area_union),0)
        return iou

    def iou_cls_ids(self,bbx_cls,bbx_ids,iou_threshold):
        '''calculate the iou of the bbox from yolov4 and deepsort'''
        match_list = []
        for cls in bbx_cls:
            area_cls = (cls[2]-cls[0]) * (cls[3]-cls[1])
            for ids in bbx_ids:
                iou_tl = [np.maximum(cls[0],ids[0]),np.maximum(cls[1],ids[1])]
                iou_br = [np.minimum(cls[2],ids[2]),np.minimum(cls[3],ids[3])]
                iou_w = np.maximum((iou_br[0] - iou_tl[0]) , 0)
                iou_h = np.maximum((iou_br[1] - iou_tl[1]) , 0)
                area_intersection = iou_w * iou_h
                area_ids = (ids[2]-ids[0]) * (ids[3]-ids[1])
                area_union = (area_cls + area_ids - area_intersection)
                iou = np.maximum((area_intersection / area_union),0)
                # print("iou in iou_cls_ids",iou)
                if iou >= iou_threshold:
                    match_list.append([ids[0],ids[1],ids[2],ids[3],ids[4],cls[4]])
                    break
        return match_list

    def cal_gredient_matrix(self,frame):
        height, width = frame.shape[:2]
        points1 = np.float32([[281,250],[467,253],[226,310],[443,310]])
        # points1 = np.float32([[457,89],[628,79],[91,410],[893,348]])
        points2 = np.float32([[0,0],[int(width),0],[0,height],[int(width),height]])
        M = cv2.getPerspectiveTransform(points1,points2)
        gredient_matrix = np.zeros((width,height))
        print(height,width)
        for i in range(width-1):
            for j in range(height-1):
             result =  np.dot(M,np.array(([i],[j],[1])))
             x = result[0]/result[2]
             y = result[1]/result[2]
             result_2 =  np.dot(M,np.array(([i-4],[j+5],[1])))
             x_2 = result_2[0]/result_2[2]
             y_2 = result_2[1]/result_2[2]
             gredient_matrix[i][j] = np.sqrt(((x-x_2)*(6/400.0))**2 + ((y-y_2)*(4/425.0))**2)  *(np.sqrt(2) /4)* 0.5
        return gredient_matrix

    def run(self):
        idx_frame = 0
        downward_counter = [0,0,0,0,0]
        upward_counter = [0,0,0,0,0]
        memory = []
        vehicle_history = []
        transform_vehicle_history = []
        yolo_bboxes = [] 
        class_names = load_class_names( 'obj.names')
        while self.vdo.grab():
            idx_frame += 1
            # if idx_frame % self.args.frame_interval:
            #     continue
            print("===========Start a frame===========")
            start = time.time()
            frame_time1 = time.time()
            ret , ori_im = self.vdo.retrieve()
            frame_time2 = time.time()
            frame_time = frame_time2 - frame_time1
            print("frame_time : ", frame_time)
            print("idx_frame : ", idx_frame)
            im = self.sat(ori_im)
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            out_img = copy.deepcopy(ori_im)
            
            '''=====depend on data====='''
            upward_line_pos = 350
            downward_line_pos = 410
            line = [(190, 317), (900, 317)]
            #0715counting lines
            # out_img = cv2.line(ori_im,(100,downward_line_pos),(400,downward_line_pos),(255,0,0),3,cv2.LINE_AA)
            # out_img = cv2.line(ori_im,(590,upward_line_pos),(890,upward_line_pos),(255,0,0),3,cv2.LINE_AA)
            #0715downway
            # cv2.fillPoly(ori_im,np.array([[(180,330),(430,330),(412,370),(135,370)]]),(0,255,0)) 
            # cv2.fillPoly(ori_im,np.array([[(135,370),(412,370),(395,410),(90,410)]]),(255,0,0))
            #0715 upperway
            # cv2.fillPoly(ori_im,np.array([[(581,299),(838,299),(890,347),(585,343)]]),(0,255,0)) 
            # cv2.fillPoly(ori_im,np.array([[(585,343),(890,347),(899,428),(586,428)]]),(255,0,0)) 
            #tdx_highway lane changing
            cv2.fillPoly(ori_im,np.array([[(23,432),(215,215),(267,160),(287,159),(258,216),(178,408)]]),(180,80,180)) 
            cv2.fillPoly(ori_im,np.array([[(178,408),(258,216),(287,159),(310,154),(304,214),(303,408)]]),(160,160,60)) 
            cv2.fillPoly(ori_im,np.array([[(310,154),(304,214),(303,408),(439,443),(329,156)]]),(70,170,170)) 
            cv2.fillPoly(ori_im,np.array([[(2,343),(234,167),(267,160),(215,215),(23,432)]]),(0,125,100)) 

            #tdx_highway downway
            cv2.fillPoly(ori_im,np.array([[(25,326),(397,340),(419,399),(2,343)]]),(255,0,0))
            # cv2.fillPoly(ori_im,np.array([[(164,215),(347,212),(397,340),(25,326)]]),(255,0,0)) 

            # cv2.fillPoly(out_img,np.array([[(23,432),(215,215),(267,160),(287,159),(258,216),(178,408)]]),(120,120,0)) 

            '''=====get detection from yolov4_pytorch====='''
            height, width = im.shape[:2]
            sized=cv2.resize(im,(self.detector.width,self.detector.height))
            boxes = do_detect(self.detector, sized, conf_thresh = 0.2, nms_thresh = 0.5, num_classes = 5 , use_cuda = 1)
            # draw_img = plot_boxes_cv2(ori_im, boxes,class_names=class_names)

            bbx_cls = self.bbox_transform(boxes,ori_im)
            bbx_cls = np.array(bbx_cls)
            print("bbx_cls\n",bbx_cls)
            # for i,bbox in enumerate(bbx_cls):
            #     cv2.rectangle(ori_im,(bbox[0], bbox[1]),(bbox[2],bbox[3]),(200,124,86),3)

            yolo_bbox_cls = []
            '''=====yolov4 bbox buffer====='''
            if len(yolo_bboxes) == 0:
                for i in bbx_cls:
                    yolo_bboxes.append(my_vehicle.his_yolo_bbox([i[0],i[1],i[2],i[3],i[4]]))
            if idx_frame != 1:
                for i,bbox_cls in enumerate (bbx_cls):
                    for j,yolo_bbox in enumerate (yolo_bboxes):
                        if yolo_bbox.matched == 1:
                            continue
                        else:
                            iou = self.iou_bbox(bbox_cls,yolo_bbox.bbox)
                            if iou > 0.3:
                                yolo_bbox.update(bbox_cls)
                                break
                            else:
                                if j == len(yolo_bboxes)-1:
                                    yolo_bboxes.append(my_vehicle.his_yolo_bbox([bbox_cls[0],bbox_cls[1],bbox_cls[2],bbox_cls[3],bbox_cls[4]]))
                                else:
                                    yolo_bbox.not_match()
                for index, his in enumerate (yolo_bboxes):
                    if his.matched == 0:
                        his.not_match()
                        if his.yolo_age > 5:
                            del yolo_bboxes[index]
                    else:
                        his.matched = 0
                        yolo_bbox_cls.append([his.bbox[0],his.bbox[1],his.bbox[2],his.bbox[3],his.bbox[4]])
                        # cv2.rectangle(ori_im,(his.bbox[0], his.bbox[1]),(his.bbox[2],his.bbox[3]),(200,124,86),3)
                yolo_bbox_cls = np.array(yolo_bbox_cls)
            # ori_im = plot_boxes_cv2(ori_im, yolo_bbox_cls,class_names=class_names)

            if len(boxes) == 0:
                print("===No detection===")
                continue

            '''=====data to the deepsort====='''
            boxes = torch.tensor(boxes)
            bbox = boxes[:, :4]
            bbox_cxcywh = bbox * torch.FloatTensor([[width, height, width, height]])
            cls_conf = boxes[:, 5]
            if bbox_cxcywh is not None:
                bbox_cxcywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                
            '''=====do tracking x1 y1 x2 y2 via deepsort====='''

            bbx_ids = self.deepsort.update(bbox_cxcywh, cls_conf, im)
            
            '''=====draw deepsort boxes for visualization====='''
            # if len(bbx_ids) > 0:
            #     print("bbx_ids\n",bbx_ids)
            #     print("============")
            #     bbox_xyxy = bbx_ids[:, :4]
            #     identities = bbx_ids[:, -1]
            #     ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

            '''=====count the iou of two bbx to find the match id with class====='''
            match_list = self.iou_cls_ids(yolo_bbox_cls,bbx_ids,iou_threshold=0.3)
            # match_list = np.array(match_list)
            print("match_list\n",match_list)

            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
            current_box = []
            previous_box = copy.deepcopy(memory)
            indexIDs = []
            memory = []
            bbx = []

            '''=====calculating the gredient of the image at the first frame====='''
            # if  idx_frame == 1:
            #     gredient_matrix = self.cal_gredient_matrix(ori_im)

            '''=====extract data from match list====='''
            if len(match_list) :
                for match in match_list:
                    center_color = ori_im[int((match[3]))-2,int((match[0]+match[2])/2)]
                    current_box.append(my_vehicle.Vehicle(match[0], match[1],match[2], match[3],match[4],match[5],center_color))
                    indexIDs.append(int(match[4]))
                    bbx.append([match[0], match[1],match[2], match[3]])
                    memory.append(my_vehicle.Vehicle(match[0], match[1],match[2], match[3],match[4],match[5],center_color))

                '''=====draw the trajectory of the vehicle with same ids====='''
                if len(vehicle_history) == 0:
                    for new_v in match_list:
                        center_color = ori_im[int((new_v[3]+new_v[1])/2)+2,int((new_v[0]+new_v[2])/2)]
                        vehicle_history.append(my_vehicle.Vehicle(new_v[0],new_v[1],new_v[2],new_v[3],new_v[4],new_v[5],center_color))
                else:
                    for i,old_v in enumerate(vehicle_history):
                        for j,new_v in enumerate(match_list):
                            center_color = ori_im[int((new_v[3]+new_v[1])/2)+2,int((new_v[0]+new_v[2])/2)]
                            if new_v[4] == old_v.ids:
                                if (old_v.color == [180,80,180]).all() and (center_color == ([160,160,60])).all():
                                    old_v.lane_change()
                                if (old_v.color == [160,160,60]).all() :
                                    if (center_color == ([180,80,180])).all() or (center_color == ([70,170,170])).all():
                                        old_v.lane_change()
                                if (old_v.color == [83,183,183]).all() and (center_color == ([160,160,60])).all():
                                    old_v.lane_change()
                                if ((new_v[1]+new_v[3])/2)-(old_v.y1+old_v.y2)/2 < 0:
                                    cv2.putText(out_img, "Wrong way driving", (old_v.x1, old_v.his_cy[-1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (120, 120, 255),2)
                                    
                                if len(old_v.his_cx) > 30 :
                                    del old_v.his_cx[0]
                                    del old_v.his_cy[0]
                                old_v.update(new_v[0],new_v[1],new_v[2],new_v[3],new_v[4],center_color)
                                del match_list[j]
                                break
                            else:
                                if j == len(match_list)-1:
                                    old_v.count_age()
                        if old_v.age > 10:
                            del vehicle_history[i]
                        if old_v.lane_age :
                            print('the car :',old_v.ids)
                            cv2.putText(out_img, "Lane Changing", (old_v.x2, old_v.y2), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 120, 0),2)
                            old_v.lane_age -=1
                    if len(match_list):
                        for new_v in match_list:
                            center_color = ori_im[int((new_v[3]+new_v[1])/2)+2,int((new_v[0]+new_v[2])/2)]
                            vehicle_history.append(my_vehicle.Vehicle(new_v[0],new_v[1],new_v[2],new_v[3],new_v[4],new_v[5],center_color))

                    '''=====calculating the speed via the pixel and the gredient====='''
                    # for old_v in vehicle_history:
                    #     for i in range(len(old_v.his_cx)-1):
                    #         ori_im = cv2.line(ori_im,(old_v.his_cx[i],old_v.his_cy[i]),(old_v.his_cx[i+1],old_v.his_cy[i+1]),(128,128,0),3,cv2.LINE_8)
                    #         ori_im = cv2.circle(ori_im,(old_v.his_cx[i],old_v.his_cy[i]),2,(0,0,255),cv2.FILLED)
                    #     if len(old_v.his_cx) > 1  :
                    #         # print("gredient_matrix_shape",shape(gredient_matrix))
                    #         dis = ((np.sqrt(((old_v.his_cx[-2] - old_v.his_cx[-2])**2+(old_v.his_cy[-1] - old_v.his_cy[-1])**2))) * \
                    #         ((gredient_matrix[old_v.his_cx[-1]][old_v.his_cy[-1]]+gredient_matrix[old_v.his_cx[-2]][old_v.his_cy[-2]])/2.0))

                    #         pixel_dis = (np.sqrt(((old_v.his_cx[-2] - old_v.his_cx[-1])**2+(old_v.his_cy[-2] - old_v.his_cy[-1])**2)))
                    #         gredient = ((gredient_matrix[old_v.his_cx[-1]][old_v.his_cy[-1]]+gredient_matrix[old_v.his_cx[-2]][old_v.his_cy[-2]])/2.0)
                    #         real_dis = np.multiply(pixel_dis,gredient)
                    #         speed = real_dis/ 0.03333 *3.6
                    #         avg_speed = old_v.write_speed(speed)
                    #         ori_im = cv2.putText(ori_im,str(int(avg_speed)),(old_v.x2,old_v.y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    #         print("current cars : ")


            # current_box = np.array(current_box,int)
            # bbx = np.array(bbx,int)
            # print("current_box\n",current_box)
            # print("====================")
            # previous_box = np.array(previous_box,int)
            # print("previous\n",previous_box)
            # print("====================")
            # buttom_center = np.array(buttom_center,int)
            # print("buttom center\n",buttom_center)
            # print("====================")

            '''=====draw the bbox for visualization====='''
            draw_img = draw_boxes(out_img,bbx,indexIDs)

            '''=====data processing  class and  counting===== '''
            if len(current_box) > 0:
                i = int(0)
                for cur_box in current_box:
                    # cur_x1,cur_y1,cur_x2,cur_y2 = cur_box[0], cur_box[1],cur_box[2],cur_box[3]
                    cur_x1,cur_y1,cur_x2,cur_y2 = cur_box.x1, cur_box.y1,cur_box.x2,cur_box.y2
                    cur_cx,cur_cy = int((cur_x1+cur_x2)/2),int((cur_y1+cur_y2)/2)
                    for pre_bbx in previous_box:
                        # print("previous color : ", pre_bbx.color)
                        # pre_x1,pre_y1,pre_x2,pre_y2 = pre_bbx[0], pre_bbx[1],pre_bbx[2],pre_bbx[3]
                        pre_x1,pre_y1,pre_x2,pre_y2 = pre_bbx.x1, pre_bbx.y1,pre_bbx.x2,pre_bbx.y2
                        pre_cx,pre_cy = int((pre_x1+pre_x2)/2),int((pre_y1+pre_y2)/2)
                        if (cur_box.ids == pre_bbx.ids) or self.cal_dist(cur_cx,cur_cy,pre_cx,pre_cy) < 30:
                            if (pre_bbx.color == [180,80,180]).all() or (pre_bbx.color == [160,160,60]).all() or (pre_bbx.color == [70,170,170]).all() or (pre_bbx.color == [0,125,100]).all():
                                if (cur_box.color == [255,0,0]).all():
                                    if cur_box.classes == 0:
                                        downward_counter[0] +=1
                                    elif cur_box.classes == 1:
                                        downward_counter[1] += 1
                                    elif cur_box.classes == 2:
                                        downward_counter[2] += 1
                                    elif cur_box.classes == 3:
                                        downward_counter[3] += 1
                                    elif cur_box.classes == 4:
                                        downward_counter[4] += 1
                            elif(pre_bbx.color == [255,0,0]).all():
                                if (cur_box.color == [0,255,0]).all():
                                    if cur_box.classes == 0:
                                        upward_counter[0] +=1
                                    elif cur_box.classes == 1:
                                        upward_counter[1] += 1
                                    elif cur_box.classes == 2:
                                        upward_counter[2] += 1
                                    elif cur_box.classes == 3:
                                        upward_counter[3] += 1
                                    elif cur_box.classes == 4:
                                        upward_counter[4] += 1


                        # if (cur_box[4] == pre_bbx[4]) or self.cal_dist(cur_cx,cur_cy,pre_cx,pre_cy) < 30:
                            # print("cal_dis",self.cal_dist(cur_cx,cur_cy,pre_cx,pre_cy))
                            # print("same vehcile")
                            # if pre_y1 < downward_line_pos and pre_y2 <=downward_line_pos:
                            #     if cur_y1 < downward_line_pos and cur_y2 >= downward_line_pos:
                            #         if cur_box[5] == 0:
                            #             downward_counter[0] +=1
                            #         elif cur_box[5] == 1:
                            #             downward_counter[1] += 1
                            #         elif cur_box[5] == 2:
                            #             downward_counter[2] += 1
                            #         elif cur_box[5] == 3:
                            #             downward_counter[3] += 1
                            #         elif cur_box[5] == 4:
                            #             downward_counter[4] += 1
                            # elif pre_y1 > upward_line_pos and pre_y2 > upward_line_pos:
                            #     if cur_y2 > upward_line_pos and cur_y1 < upward_line_pos:
                            #         if cur_box[5] == 0:
                            #             upward_counter[0] +=1
                            #         elif cur_box[5] == 1:
                            #             upward_counter[1] += 1
                            #         elif cur_box[5] == 2:
                            #             upward_counter[2] += 1
                            #         elif cur_box[5] == 3:
                            #             upward_counter[3] += 1
                            #         elif cur_box[5] == 4:
                            #             upward_counter[4] += 1
                                        
            end = time.time()
            processed_frame_time = end - start
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / processed_frame_time))
            print("===========End of a frame===========")

            '''=====左邊文字背景框====='''
            cv2.rectangle(draw_img, (0, 0), (160, 80), (85, 0, 0), -1)
            cv2.putText(draw_img, "Car : " + str(downward_counter[1]), (0, 30), cv2.FONT_HERSHEY_COMPLEX, .6, (170, 255, 50),1)
            cv2.putText(draw_img, "motorcycle : " + str(downward_counter[2]), (0, 50), cv2.FONT_HERSHEY_COMPLEX, .6, (50, 255, 50),1)
            cv2.putText(draw_img, "truck : " + str(downward_counter[4]), (0, 70), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 250, 150),1)
            '''=====右邊文字背景框====='''
            cv2.rectangle(draw_img, (width-160, 0), (width, 80), (85, 0, 0), -1)
            cv2.putText(draw_img, "Car : " + str(upward_counter[1]), (width-160, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
            cv2.putText(draw_img, "motorcycle : " + str(upward_counter[2]), (width-160, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
            cv2.putText(draw_img, "truck : " + str(upward_counter[4]), (width-160, 70), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
            print(downward_counter[1],downward_counter[2],downward_counter[4])
            if self.args.display:
                cv2.imshow("test", draw_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.vdo.release()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--cfg_detection", type=str, default="./configs/yolov4_obj.yaml")
    # yolov4.cfg/yolov4.weights/yolo-obj.cfg/yolo-obj_best.weights
    parser.add_argument("--cfg_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=1080)
    parser.add_argument("--display_height", type=int, default=720)
    # parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    cfg=get_config()
    cfg.merge_from_file(args.cfg_detection)
    cfg.merge_from_file(args.cfg_deepsort)
    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()