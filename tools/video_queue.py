import ipdb;pdb=ipdb.set_trace
import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
main_path = os.path.join(path, '..')
sys.path.insert(0, main_path)
import time
import numpy as np
from pipeline import VideoLoader, Detector, PoseProcessor
from common.arguments import parse_args
from common.model import TemporalModel
import torch


args = parse_args()

def get_3Dmodel():
    # model loads
    model_pos = TemporalModel(17, 2, 17,filter_widths=[3,3,3,3,3] , causal=args.causal, dropout=args.dropout, channels=args.channels, dense=args.dense)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    # load trained model
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', os.path.join(main_path, chk_filename))
    checkpoint = torch.load(os.path.join(main_path,chk_filename), map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    return model_pos

def get_2Dmodel():
    from joints_detectors.openpose.main import load_model
    joint_model = load_model()
    return joint_model


def main():
    model_2D = get_2Dmodel()
    model_3D = get_3Dmodel()
    dataloader = VideoLoader(args.viz_video)
    detector = Detector(dataloader, model_2D, model_3D)
    pose_processor = PoseProcessor(detector, queue_size=1000)

    dataloader.start()
    detector.start()
    pose_processor.start()
    pdb()
    while not pose_processor.stopped:
        try:
            img = pose_processor.next(timeout=1)
        except:
            continue

if __name__ == '__main__':
    main()
    print("OVER")
