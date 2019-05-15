import sys
import os
import time
import tqdm
import threading
import ipdb;pdb=ipdb.set_trace
import numpy as np
import cv2
import queue
import requests
from joints_detectors.openpose.main import generate_frame_kpt
from common.generators import ChunkedGenerator, UnchunkedGenerator
import time
from common.visualization import render_animation, render_animation_test
from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, camera_to_world, image_coordinates
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2



args = parse_args()
metadata={'layout_name': 'coco','num_joints': 17,'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16]]}
keypoints_symmetry = metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
rot = np.array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=np.float32)

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






def draw_image(keypoints, anim_output, image):
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

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()
    pos = anim_output

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
    plt.close(fig)
    return image











class PoseStage:
    def __init__(self, target, queue_size=None, prev=None):
        self.target = target
        self.stopped = False
        self.prev = prev
        # initialize the queue used to store data
        if queue_size is not None:
            self.Q = queue.Queue(maxsize=queue_size)
        self.t = None

    def next(self, timeout=None):
        return self.Q.get(timeout=timeout)

    def start(self):
        self.t = threading.Thread(target=self.target, args=())
        self.t.daemon = True
        self.t.start()

    def wait_for_queue(self, time_step):
        while self.Q.full():
            time.sleep(time_step)

    def wait_for_stop(self, time_step):
        if self.prev is not None:
            while not self.prev.stopped:
                time.sleep(time_step)
        while not self.Q.empty():
            time.sleep(time_step)


class VideoLoader(PoseStage):
    '''Load video for prediction'''
    def __init__(self, path, queue_size=16):
        super(VideoLoader, self).__init__(self.get_image, queue_size)
        self.path = path
        self.stream = cv2.VideoCapture(path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.data_len = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.data_len

    def video_info(self):
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frame_size = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return fourcc, fps, frame_size

    def get_image(self):
        time_rec = []
        for i in range(self.data_len):
            tic = time.time()
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stream.release()
                self.wait_for_stop(1)
                print('VideoLoader: %fs' % (np.mean(time_rec)))
                self.stopped = True
                return

            toc = time.time()
            time_rec.append(toc - tic)

            # put into queue
            self.wait_for_queue(0.5)
            self.Q.put(frame)

        self.wait_for_stop(1)
        print('VideoLoader: %fs' % (np.mean(time_rec)))
        self.stopped = True


class Detector(PoseStage):
    '''2D joints and 3D pose detection'''
    def __init__(self, data_loader, model_2D, model_3D, queue_size=16):
        super(Detector, self).__init__(self.detect_fn, queue_size, prev=data_loader)
        self.data_len = len(data_loader)
        self.data_queue = []
        self.model_2D = model_2D
        self.model_3D = model_3D

    def __len__(self):
        return self.data_len

    def detect_fn(self):
        time_rec = []
        for _ in tqdm.tqdm(range(self.data_len)):
            if self.prev.stopped:
                break

            try:
                img = self.prev.next(timeout=1)
                print('get image which shape is ', img.shape)
            except Exception as e:
                print(e)
                continue

            tic = time.time()
            if len(self.data_queue) < 20:
                for k in range(20):
                    kpt = generate_frame_kpt(img, self.model_2D)
                    img = self.prev.next(timeout=1)
                    self.data_queue.append(kpt)

            if len(self.data_queue) > 180:
                self.data_queue.pop(0)
                kpt = generate_frame_kpt(img, model_2D)
                self.data_queue.append(kpt)

            keypoints = np.array(self.data_queue)
            if args.causal:
                print('INFO: Using causal convolutions')
                causal_shift = pad
            else:
                causal_shift = 0
            receptive_field = self.model_3D.receptive_field()
            pad = (receptive_field - 1) // 2 # Padding on each side

            keypoints = normalize_screen_coordinates(keypoints[..., :2],
                    w=1000, h=1002)
            input_keypoints = keypoints.copy()
            gen = UnchunkedGenerator(None, None, [input_keypoints],
                                        pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            prediction = evaluate(gen, self.model_3D, return_predictions=True)
            prediction = camera_to_world(prediction, R=rot, t=0)

            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            anim_output = prediction
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

            toc = time.time()
            time_rec.append(toc - tic)
            self.wait_for_queue(0.1)
            self.Q.put((input_keypoints, anim_output, img))

        self.wait_for_stop(1)
        print('Detector: %fs' % (np.mean(time_rec)))
        self.stopped = True


class PoseProcessor(PoseStage):
    def __init__(self, detector, queue_size=1024):
        super(PoseProcessor, self).__init__(self.process, queue_size, prev=detector)
        self.data_len = len(detector)
        data_queue = []

    def __len__(self):
        return self.data_len

    def process(self):
        time_rec = []
        im_names_desc = range(self.data_len)
        for _ in im_names_desc:
            if self.prev.stopped:
                break

            try:
                keypoints, anim_output, img = self.prev.next(timeout=1)
                print('get 3D model detection results ')
            except Exception as e:
                print(e)
                continue

            tic = time.time()

            draw_image(keypoints, anim_output, img)

            toc = time.time()
            time_rec.append(toc - tic)
            #  self.Q.put(image)

        # self.wait_for_stop(1)
        while not self.prev.stopped:
            time.sleep(1)
        print('PoseProcessor: %fs' % (np.mean(time_rec)))
        self.stopped = True
