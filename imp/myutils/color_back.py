import copy
import cv2
import os
import json
from tqdm import tqdm 
root = "/home/data/p2/38.29/images"
outp = "/home/data/p2/38.29/color_back"
core_bbox = "/root/autodl-tmp/orimg/corse_bbox"

  
os.makedirs(outp, exist_ok=True)

def box_color_back():
    filelist = os.listdir(root)
    for fname in tqdm(filelist):
        fpth = os.path.join(root, fname)
        im0 = cv2.imread(fpth)

        boexs_labels = json.load(open(os.path.join(core_bbox,fname.replace(".jpg", ".json")), "r"))["shapes"]

        boxes = []
        clss = []
        for idx in range(len(boexs_labels)):
            bb = [item for sublist in boexs_labels[idx]["points"] for item in sublist]
            boxes.append(bb)
            clss.append(boexs_labels[idx]["label"])

        
        for box, cls in zip(boxes, clss):
            obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            R = copy.deepcopy(obj[:,:,2])
            G = copy.deepcopy(obj[:,:,1])
            B = copy.deepcopy(obj[:,:,0])
            # 1. COLOR_BGR2RGB
            obj[:,:,0] = R 
            obj[:,:,1] = G 
            obj[:,:,2] = B 

            im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = obj

        cv2.imwrite(os.path.join(outp, fname), im0)

def whole_color_back():
    filelist = os.listdir(root)
    for fname in tqdm(filelist):
        fpth = os.path.join(root, fname)
        im0 = cv2.imread(fpth)

        G = copy.deepcopy(im0[:,:,2])
        B = copy.deepcopy(im0[:,:,1])
        R = copy.deepcopy(im0[:,:,0])
        # 1. COLOR_BGR2RGB
        im0[:,:,0] = B 
        im0[:,:,1] = G 
        im0[:,:,2] = R 
        
        cv2.imwrite(os.path.join(outp, fname), im0)