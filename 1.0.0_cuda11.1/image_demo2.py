# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) University of Waikato, Hamilton, NZ
import cv2
import numpy as np

from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--output_file', required=True, help='The generated segmentation')
    parser.add_argument('--prediction_format', choices=["grayscale", "bluechannel"], default="grayscale", help='How to output the segmentation')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # test a single image
    result = inference_model(model, args.img)
    pr_mask = result[0]
    pr_mask = np.array(pr_mask, dtype=np.uint8)

    # not grayscale?
    if args.prediction_format == "bluechannel":
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
        pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
        pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])

    # save segmentation
    cv2.imwrite(args.output_file, pr_mask)


if __name__ == '__main__':
    main()
