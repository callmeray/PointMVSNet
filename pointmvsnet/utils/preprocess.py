import numpy as np
import cv2
import math


def norm_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 1e-7)


def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam


def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def scale_dtu_input(images, cams, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    for view in range(len(images)):
        images[view] = scale_image(images[view], scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)

    if depth_image is None:
        return images, cams
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        return images, cams, depth_image


def crop_dtu_input(images, cams, height, width, base_image_size, depth_image=None):
    """ resize images and cameras to fit the network (can be divided by base image size) """

    # crop images and cameras
    for view in range(len(images)):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > height:
            new_h = height
        else:
            new_h = int(math.floor(h / base_image_size) * base_image_size)
        if new_w > width:
            new_w = width
        else:
            new_w = int(math.floor(w / base_image_size) * base_image_size)
        start_h = int(math.floor((h - new_h) / 2))
        start_w = int(math.floor((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams
