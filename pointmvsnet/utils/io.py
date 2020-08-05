import re
import numpy as np
import sys


def load_cam_dtu(file_path, num_depth, interval_scale=1.0):
    """ read camera txt file """
    file = open(file_path)
    words = file.read().split()
    assert len(words) in [29, 30, 31], "Wrong format for camera file"
    # cam: [[[R t]
    #       [0 1]],
    #       [[K 0]
    #       [depth_start, depth_interval, num_depth, depth_end]]]
    cam = np.zeros((2, 4, 4))
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = num_depth
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 30:
        depth_start = float(words[27])
        origin_depth_interval = float(words[28])
        origin_num_depth = int(words[29])
        depth_end = depth_start + (origin_num_depth - 1) * origin_depth_interval
        depth_interval = (depth_end - depth_start) / (num_depth - 1)
        cam[1][3][0] = depth_start
        cam[1][3][1] = depth_interval
        cam[1][3][2] = num_depth
        cam[1][3][3] = depth_end
    else:
        depth_start = float(words[27])
        depth_end = float(words[30])
        depth_interval = (depth_end - depth_start) / (num_depth - 1)
        cam[1][3][0] = depth_start
        cam[1][3][1] = depth_interval
        cam[1][3][2] = num_depth
        cam[1][3][3] = depth_end

    file.close()
    return cam


def write_cam(file, cam):
    f = open(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

    return


def load_pfm(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()

    return
