# Save / show debug images
# Written by Mark Strefford, Delirium Digital Limited

import cv2
import math
import numpy as np


def render_debug_image(cfg, height, width, batch_image, pred):
    # Render marker for position (x,y for now)
    # rotation = y[3:].numpy()  # TODO: Render rotation

    # TODO: This only works as width/2 and height/2 are both 128!!
    pred = np.squeeze(pred) * 128  # TODO: Range is -1 to +1, needs to be -128 to +128
    imgarr = (np.array(batch_image) + 1) * 127  # Denormalize colour range. TODO: Merge with cvtColor below
    # for k in range(cfg.DATASET.NUM_KEYPOINTS):
    # TODO: Handle config in GA code
    for k in range(5):
        kp_idx = k * 3  # TODO: Tidy up!! cfg.DATASET.NUM_FEATURES_PER_KEYPOINT  # (x, y, z, visible)
        keypoint = np.array(pred[kp_idx:kp_idx + 2])  # Just using x and y for now
        keypoint[0] = keypoint[0] + (width / 2)  # x=0 is image center!!
        keypoint[1] = -keypoint[1] + (height / 2)  # y=0 is image center!!  three.js +y is up, opencv +y is down
        # print(keypoint[0], keypoint[1])
        color = [0, 0, 255] if kp_idx == 0 else [255, 255, 0]
        cv2.circle(imgarr, (int(keypoint[0]), int(keypoint[1])), 2, color, 2)
        imgarr = cv2.cvtColor(imgarr, cv2.COLOR_BGR2RGB)
    return imgarr


def save_debug_images(cfg, batch_images, meta, target, preds, prefix):
    # if not cfg.DEBUG.DEBUG:
    #     return

    # TODO: Need to render ground truth too!

    # Create blank canvas for batch_images
    nrow = 8
    padding = 2

    # TODO: Remove extra padding in code below
    nmaps = batch_images.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_images.shape[1] + padding)
    width = int(batch_images.shape[2] + padding)
    ndarr_pred = np.zeros((ymaps * height + padding, xmaps * width + padding, 3), dtype=np.float32)
    ndarr_output = np.zeros_like(ndarr_pred)
    # Render marker for position (x,y for now)
    x_idx = 0
    for j in range(ymaps):
        for x in range(xmaps):

            img = batch_images[x_idx]  # .numpy()

            # print(f'preds: {preds[x_idx].shape}')
            if preds:
                pred_img = render_debug_image(cfg, height, width, img, preds[0][x_idx])  # TODO: Coords only here
                ndarr_pred[j * width:(j + 1) * width - padding, x * width:(x + 1) * width - padding, :] = pred_img

            # print(f'target: {target[x_idx].shape}')
            output_img = render_debug_image(cfg, height, width, img, target['coords'][x_idx])
            ndarr_output[j * width:(j + 1) * width - padding, x * width:(x + 1) * width - padding, :] = output_img

            x_idx += 1
            if x_idx > nmaps:
                break

    if preds:
        cv2.imwrite(f'{prefix}_pred.jpg', ndarr_pred)
    cv2.imwrite(f'{prefix}_gt.jpg', ndarr_output)
