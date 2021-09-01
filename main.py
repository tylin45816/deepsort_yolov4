import argparse
from math import inf
from utils.sort.iou_matching import iou
from numpy.core.fromnumeric import shape
import torch
import cv2
import os
import time
import numpy as np
import random
import copy
from utils.utils import get_config, do_detect,plot_boxes_cv2,load_class_names
from utils.draw import draw_boxes
from utils import build_detector, build_deepsort

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
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
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
        output = []
        wh = np.flip(im.shape[0:2])
        for class_index in range(len(boxes)-1):
            cxcy = tuple((np.array(boxes[class_index][0:2]) * wh).astype(np.int64))
            bbx_wh = tuple((np.array(boxes[class_index][2:4]) * wh).astype(np.int64))
            cx, cy = cxcy
            w, h = bbx_wh
            x1 = int(abs(cx-w/2))
            y1 = int(abs(cy-h/2)*0.97)
            x2 = int(cx+w/2)
            y2 = int((cy+h/2)*1.03)
            classes = int(boxes[class_index][6])
            if x1 < 0:
                continue
            
            im = cv2.circle(im,((cx),(cy)),2,(0,0,255),cv2.FILLED)
            im = cv2.circle(im,((x1),(y1)),3,(255,0,255),cv2.FILLED)
            im = cv2.circle(im,((x2),(y2)),3,(0,255,255),cv2.FILLED)
            
            output.append([x1,y1,x2,y2,classes])
        return output

    def iou_cls_ids(self,bbx_cls,bbx_ids,iou_threshold):
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
                print("iou in iou_cls_ids",iou)
                if iou > iou_threshold:
                    match_list.append([ids[0],ids[1],ids[2],ids[3],ids[4],cls[4]])
                    break
        return match_list

    def run(self):
        idx_frame = 0
        downward_counter = [0,0,0,0,0]
        upward_counter = [0,0,0,0,0]
        memory = []
        vehicle_history = []
        class_names = load_class_names( 'obj.names')
        
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            print("===========Start a frame===========")
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            
            """depend on data"""
            upward_line_pos = 350
            downward_line_pos = 350
            line = [(190, 317), (900, 317)]
            ori_im = cv2.line(ori_im,(190,downward_line_pos),(440,downward_line_pos),(255,0,0),3,cv2.LINE_AA)
            ori_im = cv2.line(ori_im,(590,upward_line_pos),(820,upward_line_pos),(255,0,0),3,cv2.LINE_AA)
            # line_pos = 500
            # line = [(160, line_pos), (1600, line_pos)]
            # result_img2 = cv2.line(img,(160,line_pos),(1600,line_pos),(255,0,0),3,cv2.LINE_AA)

            """do detection"""
            height, width = im.shape[:2]
            sized=cv2.resize(im,(self.detector.width,self.detector.height))
            boxes = do_detect(self.detector, sized, 0.5, 0.4, 1)
            ori_im = plot_boxes_cv2(ori_im, boxes, savename=None, class_names=class_names)

            
            bbx_cls = self.bbox_transform(boxes,ori_im)
            bbx_cls = np.array(bbx_cls)
            print("bbx_cls\n",bbx_cls)

            if len(boxes) == 0:
                print("===No detection===")
                continue
            boxes = torch.tensor(boxes)
            bbox = boxes[:, :4]
            bbox_cxcywh = bbox * torch.FloatTensor([[width, height, width, height]])
            cls_conf = boxes[:, 5]

            if bbox_cxcywh is not None:
                bbox_cxcywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                
                """do tracking"""
                bbx_ids = self.deepsort.update(bbox_cxcywh, cls_conf, im)
                
                """draw boxes for visualization"""
                if len(bbx_ids) > 0:
                    print("bbx_ids\n",bbx_ids)
                    print("============")
                    bbox_xyxy = bbx_ids[:, :4]
                    identities = bbx_ids[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                    
            """"count the iou of two bbx to find the match id with class"""
            match_list = self.iou_cls_ids(bbx_cls,bbx_ids,iou_threshold=0.7)
            match_list = np.array(match_list)
            vehicle_history.append(match_list)
            print("match_list\n",match_list)
            
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
            current_box = []
            previous_box = copy.deepcopy(memory)
            indexIDs = []
            memory = []
            bbx = []
            
            """extract data from match list"""
            for track in match_list:
                current_box.append(track)
                indexIDs.append(int(track[4]))
                bbx.append([track[0], track[1],track[2], track[3]])
                memory.append(track)
            
            current_box = np.array(current_box,int)
            bbx = np.array(bbx,int)
            print("current_box\n",current_box)
            print("====================")
            previous_box = np.array(previous_box,int)
            print("previous\n",previous_box)
            print("====================")
            
            # ori_im = draw_boxes(ori_im,bbx,indexIDs)
            
            """data processing  class and  counting """
            if len(current_box) > 0:
                i = int(0)
                for cur_box in current_box:
                    # cur_x1,cur_y1,cur_x2,cur_y2 = cur_box[0], cur_box[1],cur_box[2],cur_box[3]
                    # cur_cx,cur_cy = int((cur_x1+cur_x2)/2),int((cur_y1+cur_y2)/2)
                    cur_cx,cur_cy = int((cur_box[0]+cur_box[2])/2),int((cur_box[1]+cur_box[3])/2)
                    color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                    for pre_bbx in previous_box:
                        pre_x1,pre_y1,pre_x2,pre_y2 = pre_bbx[0], pre_bbx[1],pre_bbx[2],pre_bbx[3]
                        pre_cx,pre_cy = int((pre_x1+pre_x2)/2),int((pre_y1+pre_y2)/2)
                        if (cur_box[4] == pre_bbx[4]) or self.cal_dist(cur_cx,cur_cy,pre_cx,pre_cy)  < 50 :
                            print("cal_dis",self.cal_dist(cur_cx,cur_cy,pre_cx,pre_cy))
                            print("same vehcile")
                            if pre_cy < downward_line_pos and cur_cy > downward_line_pos:
                                if cur_box[5] == 0:
                                    downward_counter[0] +=1
                                elif cur_box[5] == 1:
                                    downward_counter[1] += 1
                                elif cur_box[5] == 2:
                                    downward_counter[2] += 1
                                elif cur_box[5] == 3:
                                    downward_counter[3] += 1
                                elif cur_box[5] == 4:
                                    downward_counter[4] += 1
                            elif pre_cy > upward_line_pos and cur_cy < upward_line_pos:
                                if cur_box[5] == 0:
                                    upward_counter[0] +=1
                                elif cur_box[5] == 1:
                                    upward_counter[1] += 1
                                elif cur_box[5] == 2:
                                    upward_counter[2] += 1
                                elif cur_box[5] == 3:
                                    upward_counter[3] += 1
                                elif cur_box[5] == 4:
                                    upward_counter[4] += 1
                                        
            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
            print("===========End of a frame===========")

            """左邊文字背景框"""
            cv2.rectangle(ori_im, (0, 0), (160, 80), (85, 0, 0), -1)
            cv2.putText(ori_im, "Car : " + str(downward_counter[1]), (0, 30), cv2.FONT_HERSHEY_COMPLEX, .6, (170, 255, 50),1)
            cv2.putText(ori_im, "motorcycle : " + str(downward_counter[2]), (0, 50), cv2.FONT_HERSHEY_COMPLEX, .6, (50, 255, 50),1)
            cv2.putText(ori_im, "truck : " + str(downward_counter[4]), (0, 70), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 250, 150),1)
            """右邊文字背景框"""
            cv2.rectangle(ori_im, (740, 0), (900, 80), (85, 0, 0), -1)
            cv2.putText(ori_im, "Car : " + str(upward_counter[1]), (740, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
            cv2.putText(ori_im, "motorcycle : " + str(upward_counter[2]), (740, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
            cv2.putText(ori_im, "truck : " + str(upward_counter[4]), (740, 70), cv2.FONT_HERSHEY_SIMPLEX, .6, (170, 255, 50),1)
        
            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        
        self.vdo.release()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--cfg_detection", type=str, default="./configs/yolov4.yaml")
    # yolov4.cfg/yolov4.weights/yolo-obj.cfg/yolo-obj_best.weights
    parser.add_argument("--cfg_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
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