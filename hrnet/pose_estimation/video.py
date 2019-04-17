'''
使用yolov3作为pose net模型的前处理
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/home/xyliu/experiments/VideoPose3D/hrnet/pose_estimation')
sys.path.insert(0, '/home/xyliu/experiments/VideoPose3D/hrnet')
import argparse
import os
import pprint
import ipdb;pdb=ipdb.set_trace
import numpy as np
from tqdm import tqdm
from utilitys import plot_keypoint, PreProcess
import time

import torch
import _init_paths
from config import cfg
import config
from config import update_config

from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
import models
from lib.detector.yolo.human_detector import main as yolo_det



kpt_queue = []
from scipy.signal import savgol_filter
def smooth_filter(kpts):
    if len(kpt_queue) < 6:
        kpt_queue.append(kpts)
        return kpts

    queue_length = len(kpt_queue)
    if queue_length == 50:
        kpt_queue.pop(0)
    kpt_queue.append(kpts)

    # transpose to shape (17, 2, num, 50) 关节点、横纵坐标、每帧人数、帧数
    transKpts = np.array(kpt_queue).transpose(1,2,3,0)

    window_length = queue_length - 1 if queue_length % 2 == 0 else queue_length - 2
    # array, window_length越大越好, polyorder
    result = savgol_filter(transKpts, window_length, 3).transpose(3, 0, 1, 2) #shape(frame_num, human_num, 17, 2)

    # 返回倒数第几帧
    return result[-3]


class get_args():
    cfg = '/home/xyliu/experiments/VideoPose3D/hrnet/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml'
    dataDir=''
    logDir=''
    modelDir=''
    opts=[]
    prevModelDir=''


##### load model
def model_load(config):
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    model_file_name  = '/home/xyliu/experiments/VideoPose3D/hrnet/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth'
    #  model_file_name  = 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def ckpt_time(t0=None, display=None):
    if not t0:
        return time.time()
    else:
        t1 = time.time()
        if display:
            print('consume {:2f} second'.format(t1-t0))
        return t1-t0, t1


###### 加载human detecotor model
from lib.detector.yolo.human_detector import load_model as yolo_model
human_model = yolo_model()

def generate_kpts(video_name):
    args = get_args()
    update_config(cfg, args)
    cam = cv2.VideoCapture(video_name)
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    ret_val, input_image = cam.read()
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)

    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    # 关键点收集
    kpts_result = []
    for i in tqdm(range(video_length-1)):

        ret_val, input_image = cam.read()

        try:
            bboxs, scores = yolo_det(input_image, human_model)
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(input_image, bboxs, scores, cfg)
        except Exception as e:
            print(e)
            continue

        with torch.no_grad():
            # compute output heatmap
            inputs = inputs[:,[2,1,0]]
            output = pose_model(inputs.cuda())
            # compute coordinate
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        # 平滑点
        preds = smooth_filter(preds)

        # 3D video pose 只支持单人
        kpts_result.append(preds[0])

    result = np.array(kpts_result)
    return result


if __name__ == '__main__':
    main()
