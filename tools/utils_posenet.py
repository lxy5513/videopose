import os
import sys

import torch
import numpy as np
path = os.path.split(os.path.realpath(__file__))[0]
posenet_path = os.path.join(path, '../joints_detectors/posenet/')
sys.path.insert(0, posenet_path)
import posenet

def getPosenetModel():
    model = posenet.load_model(101)
    model = model.cuda()
    return model

def getKptsFromImage(model, input_image):
    output_stride = model.output_stride
    with torch.no_grad():
        input_image, _, _ = posenet.utils._process_input(input_image, 1 / 3.0, output_stride)
        input_image = torch.Tensor(input_image)
        input_image = input_image.cuda()

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.15)

    #keypoint_coords *= output_scale
    joint2 = np.zeros(keypoint_coords[0].shape)
    joint2[:,0] = keypoint_coords[0][:,1].copy()
    joint2[:,1] = keypoint_coords[0][:,0].copy()
    return joint2
