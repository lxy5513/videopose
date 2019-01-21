import numpy as np
from graphpipe import remote
import ipdb;pdb = ipdb.set_trace
from utils import *
import torch


np_name = '../data/taiji.npz'
data = np.load(np_name)
kpts = data['kpts']
data = normlize(kpts, w=1000, h=1002)


metadata={'layout_name': 'coco','num_joints': 17,'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]}
keypoints_symmetry = metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

pad = (243-1) // 2
batch_2d = np.expand_dims(np.pad(data, ((pad, pad), (0,0), (0,0)), 'edge'), axis=0)
batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
batch_2d[1, :, :, 0] *= -1
batch_2d[1, :, kps_left + kps_right] = batch_2d[1, :, kps_right + kps_left]
batch_2d = batch_2d.astype(np.float32)

predicted_3d_pos = remote.execute("http://127.0.0.1:9000", batch_2d)
predicted_3d_pos = np.copy(predicted_3d_pos)
predicted_3d_pos[1, :, :, 0] *= -1
predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
prediction = np.mean(predicted_3d_pos, axis=0, keepdims=True)[0]


rot = params.rot()
from camera import camera_to_world
prediction = camera_to_world(prediction, R=rot, t=0)

# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])
anim_output = {'Reconstruction': prediction}

input_keypoints = re_normlize(data[..., :2], w=1000, h=1002)

output = 'xxxx.gif'
bitrate = 50000
limit =60
downsample = 1
size = 5
viz_video = '../videos/taiji.mp4'
skip = 0


class skeleton():
    def parents(self):
        return np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    def joints_right(self):
        return [1, 2, 3, 9, 10]

from visualization import render_animation
render_animation(input_keypoints, anim_output,
                    skeleton(), 25, bitrate, np.array(70., dtype=np.float32),output,
                    limit=limit, downsample=downsample, size=size,
                    input_video_path=viz_video, viewport=(1000, 1002),
                    input_video_skip=skip)


