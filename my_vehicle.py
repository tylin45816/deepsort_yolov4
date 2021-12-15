import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import is_distributed
from sklearn.metrics import mean_squared_error
from math import sqrt


class Vehicle(object):
    def __init__(self,x1,y1,x2,y2,ids,classes,color):
        self.age = 0
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.his_cx = [int((x1+x2)/2)]
        self.his_cy = [int((y2))]
        self.ids = ids
        self.classes = classes
        self.color = color
        self.color_changed = 0
        self.speed = 0
        self.flag_age = 0
        self.his_speed = []
        self.lane_age = 0
    
    def update(self,x1,y1,x2,y2,ids,color):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.ids = ids
        self.color = color
        self.his_cx.append(int((x1+x2)/2))
        self.his_cy.append(int((y2)))
        self.age = 0
        self.flag_age = 0
        return self

    def write_speed(self,speed):
        self.speed = speed
        if speed > 0:
            self.his_speed.append(speed)
        if len(self.his_speed) > 20 :
            self.his_speed.pop(0)
        geo_speed = np.array(self.his_speed)
        if len(geo_speed) :
            np_avg_speed = np.mean(geo_speed)
            print(np_avg_speed)
            # geo_avg_speed = np.prod(geo_speed)**(1.0/len(geo_speed))
        else:
            return 0
            avg_speed = sum(self.his_speed) / len(self.his_speed)
        return np_avg_speed
        
    def count_age(self):
        self.flag_age = 1
        self.age += 1
        # print("age of vehicle in match list : ",self.age)
    
    def lane_change (self):
        self.lane_age = 30
        
    def  coordinate_transform(matrix,ori_x,ori_y):
            matrix = np.array(matrix)
            ori_coord = np.array(([ori_x],[ori_y],[1]))
            result = np.dot(matrix,ori_coord)
            x,y = result[0]/result[2] , result[1]/result[2]
            return x,y

    def color_check(self):
        if self.color == (0,255,0).all():
            if self.color ==(255,0,255).all():
                self.color_changed = 1
        elif self.color == (255,0,255).all():
            if self.color == (0,255,0).all():
                self.color_changed = 1
        
class transform_vehicle(object):
    def __init__(self,ori_bcx,ori_bcy,ids,classes,color,matrix):
        self.b_c = [coordinate_transform(matrix,ori_bcx,ori_bcy)]
        self.ids = ids
        self.classes = classes
        self.color = color

    def update(self,b_cx,b_cy,color):
        self.b_c.append()
        self.color = color
        return self

    def  coordinate_transform(matrix,ori_x,ori_y):
        matrix = np.array(matrix)
        ori_coord = np.array(([ori_x],[ori_y],[1]))
        result = np.dot(matrix,ori_coord)
        x,y = result[0]/result[2] , result[1]/result[2]
        return x,y
    
class his_yolo_bbox(object):
    def  __init__(self,bbox):
        self.bbox = bbox
        self.yolo_age = 0
        self.matched = 0
    def update(self,bbox):
        self.bbox = bbox
        self.matched =1
        self.yolo_age =0
    def not_match(self):
        self.matched = 0
        self.yolo_age += 1