import argparse
import torch
import cv2
import os
import time
import numpy as np
import random
import copy
from utils.utils import get_config, do_detect
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

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
            
    def run(self):
        idx_frame = 0
        downward_counter = [0,0,0,0,0]
        upward_counter = [0,0,0,0,0]
        memory = []
        
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            # print(im)
            
            #depent on data
            up_linepos = 317
            line = [(190, 317), (900, 317)]
            # im = cv2.line(im,(190,up_linepos),(900,up_linepos),(255,0,0),3,cv2.LINE_AA)
            # up_linepos = 500
            # line = [(160, up_linepos), (1600, up_linepos)]
            # result_img2 = cv2.line(img,(160,up_linepos),(1600,up_linepos),(255,0,0),3,cv2.LINE_AA)

            # do detection
            # im=Image.fromarray(im)
            height, width = im.shape[:2]
            # sized = im.resize((self.m.width, self.m.height))
            sized=cv2.resize(im,(self.detector.width,self.detector.height))
            # print(sized)

            boxes = do_detect(self.detector, sized, 0.5, 0.4, 1)
            boxes = torch.tensor(boxes)
            
            bbox = boxes[:, :4]
            print(boxes)
            bbox_xywh = bbox * torch.FloatTensor([[width, height, width, height]])
            # print("torch.tensor",torch.FloatTensor([[width, height, width, height]]))
            cls_conf = boxes[:, 5]
            classes = boxes[:,6]
            print("classes",classes)

            # bbox_xywh, cls_conf, cls_ids = self.detector(im)

            if bbox_xywh is not None:
                # select person class
                # mask = cls_ids==0

                # bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                # cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                # outputs=[]
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                # print("bbox",bbox_xyxy)
                # print("id",identities)
                print("current_detection",outputs)
                
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")
            
            # current_box = []
            # previous_box = copy.deepcopy(memory)
            # indexIDs = []
            # memory = []
            # bbx = []
            # #extract data from sort
            # for track in outputs:
            #     current_box.append([track[0], track[1],track[2], track[3],track[5]])
            #     indexIDs.append(int(track[4]))
            #     bbx.append([track[0], track[1],track[2], track[3]])
            #     memory.append([track[0], track[1],track[2], track[3],track[5]])
            # current_box = np.array(current_box,int)
            # bbx = np.array(bbx,int)
            # print("current_box\n",current_box)
            # print("====================")
            # previous_box = np.array(previous_box,int)
            # print("previous\n",previous_box)
            # print("====================")
            
            # # result_img2 = draw.draw_boxes(img,bbx,indexIDs)
            
            # #data processing  class and  counting 
            # if len(current_box) > 0:
            #     i = int(0)
            #     for box in current_box:
            #         (x, y) = (int(box[0]), int(box[1]))
            #         (w, h) = (int(box[2]), int(box[3]))
            #         color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            #         for k in range(len(previous_box)):
            #             if VideoTracker.cal_dist(x,y,(previous_box[k][0]),(previous_box[k][1]))  < 50:
            #                 # print("same vehcile")
            #                 (x2, y2) = (int(previous_box[k][0]), int(previous_box[k][1]))
            #                 (w2, h2) = (int(previous_box[k][2]), int(previous_box[k][3]))
            #                 p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
            #                 p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
            #                 cv2.line(result_img2, p0, p1, color, 3)
            #                 if (y2 + (h2-y2)/2) < up_linepos:
            #                     if VideoTracker.intersect(p0, p1,line[0] ,line[1] ):
            #                         if box[4] == 0:
            #                             downward_counter[0] +=1
            #                         elif box[4] == 1:
            #                             downward_counter[1] += 1
            #                         elif box[4] == 2:
            #                             downward_counter[2] += 1
            #                         elif box[4] == 3:
            #                             downward_counter[3] += 1
            #                         elif box[4] == 4:
            #                             downward_counter[4] += 1
            #                 else:
            #                     if VideoTracker.intersect(p0, p1,line[0] ,line[1] ):
            #                         if box[4] == 0:
            #                             upward_counter[0] +=1
            #                         elif box[4] == 1:
            #                             upward_counter[1] += 1
            #                         elif box[4] == 2:
            #                             upward_counter[2] += 1
            #                         elif box[4] == 3:
            #                             upward_counter[3] += 1
            #                         elif box[4] == 4:
            #                             upward_counter[4] += 1
                                        
            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.args.save_path:
                self.writer.write(ori_im)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov4.yaml")
    # yolov4.cfg/yolov4.weights/yolo-obj.cfg/yolo-obj_best.weights
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    cfg=get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()