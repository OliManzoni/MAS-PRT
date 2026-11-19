from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils import visualizer
import torch
import numpy as np
import cv2
import pycocotools.mask as mask
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon
import pandas as pd
import math as m
from os import mkdir, path
import random
import glob
import stat
import shutil
import pickle 
from MAS.config import *

def frame_extract(video_path, frame_per_vid = 1):
    """Extract random frames for each video in pathToVid
    The frames selected are not completely random.
    Instead, the video is divided by how many frame needs to be extracted
    Each frame is then selected in each segments, allowing for a more linear extraction
    The frames are extracted in a 'frame' folder inside the video folders. 

    —————————
    Arguments 
    —————————
        pathToVid : The path to the video folder. Videos must be in MP4    
        frameperVid : The number of frame to extract per video. 
       
       """
    path_to_video = os.path.join(video_path, '*.mp4')
     #Clear target output directory if needed
    output = glob.glob(str(video_path + "/frames/")) 

    try:
        for f in output:
                os.chmod(f, stat.S_IWRITE)
                shutil.rmtree(f)
    except:
        pass

    os.makedirs(str(video_path + "/frames/"), exist_ok = True)
    #Create a sub-directory for each video, containing extracted frames
    for video in glob.glob(path_to_video):
        cap = cv2.VideoCapture(video)
        FileName = str(video_path + "/frames/" + str(path.basename(path.normpath(video)))).replace(".mp4","")
        totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            mkdir(FileName)
        except: 
            pass
        segment = (totalframecount-100) / frame_per_vid 
        for i in range(0,frame_per_vid) :
            frame_id = random.randint(round(segment*i), round(segment*(i+1))) #  /!\  :)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cv2.imwrite(FileName + "/Frame_" + str(i + 1) +".jpg", frame)
        cap.release()

def polygon_from_mask(maskedArr):
  # stolen from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
  """
  Change the binary mask to polygon COCO format
  """
  contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  segmentation = []
  valid_poly = 0
  for contour in contours:
     if contour.size >= 6:
        segmentation.append(contour.astype(float).flatten().tolist())
        valid_poly += 1  
  if valid_poly == 0:
     raise ValueError
  return segmentation

def predict_nest(detector_path, image_dir, visual = False, nest_simplification = 5):

    """

    Run inference from Detectron2 to detect the nest from each picture in image_dir.

    —————————
    Arguments 
    —————————

    detectorPath : Path to the detector
    image_dir : Path to one or more image in .jpg
    visual : display the nest polygon on matplotlib
    NestSimplification : Strength of Polygon.simplify(), see shapely.Polygon

    ——————————————
    Returned value 
    ——————————————

    Results are outputted as a dictonnary containing : 
    dict["id"]
    dict["video"] containing the image path
    dict["confidence"] the confidence score of the detection
    dict["polygon"] the polygon of the nest
    dict["area"] the area of the polygon
    dict["rawpolygon"] the result of the last inference
    """
    cfg=get_cfg()
    cfg.merge_from_file(os.path.join(detector_path,'config.yaml'))
    cfg.MODEL.WEIGHTS=os.path.join(detector_path,'model_final.pth')
    cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    predictor=DefaultPredictor(cfg)                         
    result = []
    results = {"id" : [], "video" : [], "videopath" : [] , "confidence" : [], "polygon" : [], "area" : [], 'rawpolygon' : []}
    i = 0
    if visual:
        temp = os.path.dirname(image_dir)
        visual_path = str(temp + '/NestImage')
        if not path.isdir(visual_path) :
            os.mkdir(visual_path)
    for folder in glob.glob(os.path.join(image_dir, '*')):
        vidname = str(path.basename(path.normpath(folder))).replace(".mp4","")
        mask = []
        
        for image in glob.glob(os.path.join(folder, '*.jpg')):
            inputs = cv2.imread(image)
            result = predictor(inputs)
            
            for masknb, pred_mask in enumerate(result['instances'].pred_masks.cpu()) :
                #print(result['instances'].scores.cpu().numpy()[masknb])
                area = cv2.countNonZero(pred_mask.numpy().astype("uint8"))
                if result['instances'].scores.cpu().numpy()[masknb] >= 0.8 and area >= MIN_AREA_THRESHOLD: 
                    mask.append(pred_mask.numpy().astype("uint8"))
                else: print("Ignored")

        if len(mask) != 1 and len(mask) != 0:
            for maskNb in range(1, len(mask)):
                mask[0] = np.add(mask[0], mask[maskNb])
            maskAv = np.rint(np.divide(mask[0],len(mask))).astype("uint8")
        elif len(mask) == 1:
            maskAv = mask[0]
        else : 
            results["confidence"].append(0)
            results["polygon"].append(0)
            results["area"].append(0)
            continue
        poly2 = []
        poly3 = []
        poly = polygon_from_mask(maskAv) #Creation of average polygon
        xcoord = poly[0][0::2]
        ycoord = poly[0][1::2]
        for j in range(0,len(xcoord)):
            poly2.append([xcoord[j], ycoord[j]])
        polygon = Polygon(poly2)

        poly2 = []
        poly = polygon_from_mask(mask[len(mask)-1]) #Last polygon for comparison
        xcoord = poly[0][0::2]
        ycoord = poly[0][1::2]
        for j in range(0,len(xcoord)):
            poly3.append([xcoord[j], ycoord[j]])
        polygon2 = Polygon(poly3)

        polygon3 = (polygon.simplify(nest_simplification))
        if visual == True:
            test = plt.imread(image)
            plt.imshow(test)
            plt.plot(*polygon2.exterior.xy, color = "red")
            plt.plot(*polygon.exterior.xy, color = "green")

            #plt.plot(*polygon3.exterior.xy, color = "blue")
            plt.text(30,-30, s = str("Green estimation is an average of " + str(len(mask)) + " detections"),  fontsize =12)
            plt.text(30,-10, s = str("Red estimation is last detection"),  fontsize =12)

            plt.savefig(visual_path + vidname + '.tiff')

        results["id"].append(i)
        results["video"].append(vidname)
        results["videopath"].append(str(folder))
        try:
            results["confidence"].append(result['instances'].scores.cpu().numpy()[0])
            results["polygon"].append(polygon)
            results["area"].append(polygon.area)
            results["rawpolygon"].append(polygon2)
    
        except:
            results["confidence"].append(0)
            results["polygon"].append(0)
            results["area"].append(0)
            results["rawpolygon"].append(0)
        
        i += 1 
    print('Video %s done' %(vidname))
    results['id'] = tuple(results['id'])
    results['video'] = tuple(results['video'])
    results['videopath'] = tuple(results['videopath'])
    results["confidence"] = tuple(results["confidence"])
    results['area'] = tuple(results['area'])
    results['rawpolygon'] = tuple(results['rawpolygon'])
    results['polygon'] = tuple(results['polygon'])
    filename = os.path.dirname(image_dir) 
    os.makedirs(os.path.join(filename, 'models'), exist_ok=True)
    with open(os.path.join(filename, 'models', 'nestDict.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return results