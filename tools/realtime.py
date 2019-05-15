'''
通过实现高精度的端到端3D姿态重建
'''
import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')
sys.path.insert(0, main_path)

import numpy as np
import ipdb;pdb = ipdb.set_trace
from common.arguments import parse_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import errno
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
import time
metadata={'layout_name': 'coco','num_joints': 17,'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]}


def draw_img(ax, anim_output, image, skeleton):
    skeleton = skeleton()
    parents = skeleton.parents()
    pos = anim_output[-1]
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'red' if j in skeleton.joints_right() else 'yellow'
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                                    [pos[j, 1], pos[j_parent, 1]],
                                    [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)


def draw_image(keypoints, anim_output, image, skeleton):
    fig = plt.figure(figsize=(12,6))
    #  canvas = FigureCanvas(fig)
    fig.add_subplot(121)
    plt.imshow(image)
    # 3D
    ax = fig.add_subplot(122, projection='3d')
    #  ax.view_init(elev=15., azim=azim)
    radius = 1.7
    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    skeleton = skeleton()

    parents = skeleton.parents()
    pos = anim_output[-1]

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in skeleton.joints_right() else 'yellow'
            # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                                    [pos[j, 1], pos[j_parent, 1]],
                                    [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    #  plt.savefig('test/3Dimage_{}.png'.format(1000+num))
    #  width, height = fig.get_size_inches() * fig.get_dpi()
    #  canvas.draw()       # draw the canvas, cache the renderer
    #  image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    #  cv2.imshow('im', image)
    #  cv2.waitKey(5)

# record time
def ckpt_time(ckpt=None, display=1, desc=''):
    if not ckpt:
        return time.time()
    else:
        if display:
            print(desc +' consume time {:0.4f}'.format(time.time() - float(ckpt)))
        return time.time() - float(ckpt), time.time()


class skeleton():
    def parents(self):
        return np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    def joints_right(self):
        return [1, 2, 3, 9, 10]

def evaluate(test_generator, model_pos, action=None, return_predictions=False):

    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    with torch.no_grad():
        model_pos.eval()

        N = 0

        for _, batch, batch_2d in test_generator.next_epoch():

            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

def main():

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    Axes3D = Axes3D
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    radius = 1.7
    ax2.set_xlim3d([-radius/2, radius/2])
    ax2.set_zlim3d([0, radius])
    ax2.set_ylim3d([-radius/2, radius/2])
    ax2.set_aspect('equal')
# 坐标轴刻度
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.dist = 7.5
    axes = [ax1, ax2]
    #  fig.show()
    fig.canvas.draw()
    backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in [ax1,ax2]]


    args = parse_args()
    if not args.viz_output:
        args.viz_output = 'outputs/op_result.mp4'

    # model loads
    model_pos = TemporalModel(17, 2, 17,filter_widths=[3,3,3,3,3] , causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    # load trained model
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', os.path.join(main_path,chk_filename))
    checkpoint = torch.load(os.path.join(main_path,chk_filename), map_location=lambda storage, loc: storage)# 把loc映射到storage
    model_pos.load_state_dict(checkpoint['model_pos'])

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2 # Padding on each side

    if args.causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    # 2D kpts loads or generate
    from joints_detectors.openpose.main import load_model, generate_frame_kpt
    joint_model = load_model()

    #  cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/home/xyliu/Videos/sports/taiji.mp4')
    kpts = []
    for i in tqdm(range(800)):
        _, frame = cap.read()
        if i < 20:
            kpt = generate_frame_kpt(frame, joint_model)
            kpts.append(kpt)
        else:

            #  if i % 2 == 0:
                #  continue

            t0 = ckpt_time()
            kpt = generate_frame_kpt(frame, joint_model)
            if len(kpts) > 189:
                kpts.pop(0)
            kpts.append(kpt)
            keypoints = np.array(kpts)

            ckpt, t1 = ckpt_time(t0, desc='load kpts')
            # normlization keypoints  假设use the camera parameter
            keypoints = normalize_screen_coordinates(keypoints[..., :2],
                    w=1000, h=1002)

            input_keypoints = keypoints.copy()
            gen = UnchunkedGenerator(None, None, [input_keypoints],
                                        pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            prediction = evaluate(gen,model_pos, return_predictions=True)

            rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)
            prediction = camera_to_world(prediction, R=rot, t=0)

            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            anim_output = {'Reconstruction': prediction}
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)
            ckpt, t2 = ckpt_time(t1, desc='generate 3D coordinate')


            items = enumerate(zip(axes, backgrounds), start=1)
            for j, (ax, background) in items:
                if j == 1:
                    fig.canvas.restore_region(background)
                    ax.imshow(frame)
                    fig.canvas.blit(ax.bbox)
                else:
                    time_1 = time.time()
                    fig.canvas.restore_region(background)
                    draw_img(ax, anim_output['Reconstruction'], frame, skeleton)
                    fig.canvas.blit(ax.bbox)
                    print('---------------> {:0.4f}'.format(time.time()-time_1))

            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            cv2.imshow('im', image)
            cv2.waitKey(5)
            ckpt, t3 = ckpt_time(t2, desc='plt image')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
