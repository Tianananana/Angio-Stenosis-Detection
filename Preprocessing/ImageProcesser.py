import os
import cv2
import numpy as np
from torchvision import transforms as tf

from DrawMainCurve import MainCurve

def mask2curve(mask, op_name, total_pts):
    """function to extract coordinates and bgr_mask (mask with crosses in BGR channels)"""
    frame = MainCurve(mask, op_name)

    # output origin matrix of pos
    pos_mat_origin = np.stack(frame.inds_ordered)
    # print(pos_mat_origin.shape)

    bgr_mask = frame.bgr_mask

    # int into float
    assert mask.shape[0] == mask.shape[1], 'Input mask is not a square.'
    pos_mat_origin = pos_mat_origin / mask.shape[0]

    origin_t = np.arange(pos_mat_origin.shape[0])
    t_float = origin_t * (total_pts-1) / origin_t[-1]
    t_int = np.arange(total_pts)

    # interpolate no. of pts
    pos_mat_inter = np.zeros((total_pts, 2))
    pos_mat_inter[:, 0] = np.interp(t_int, t_float, pos_mat_origin[:, 0])
    pos_mat_inter[:, 1] = np.interp(t_int, t_float, pos_mat_origin[:, 1])

    return pos_mat_inter, bgr_mask


def draw_pts(img, coord, n_pts):

    bgr_img = np.copy(img)
    pts_ = np.linspace(0, coord.shape[0]-1, num=n_pts, endpoint=True).astype(int)

    for pt in pts_:
        row = int(coord[pt, 0] * bgr_img.shape[0])
        col = int(coord[pt, 1] * bgr_img.shape[1])

        if pt == 0:
            bgr_img[row-3:row+4, col, :] = [0, 0, 255]
            bgr_img[row, col-3:col+4, :] = [0, 0, 255]
        else:
            bgr_img[row-3:row+4, col, :] = [255, 0, 0]
            bgr_img[row, col-3:col+4, :] = [255, 0, 0]
    return bgr_img


def draw_sten(img, coord):
    for c in coord:
        row = int(c[1])
        col = int(c[0])
        img[row - 3:row + 4, col, :] = [0, 0, 255]
        img[row, col - 3:col + 4, :] = [0, 0, 255]
    return img


def center_stenosis(mask):
    mask[mask != 22] = 0
    mask[mask == 22] = 255
    if np.max(mask) != 255:
        return None
    else:
        contours, hierarchy = cv2.findContours(mask, 1, 2)
        centroid = []
        for c in contours:
            c = np.squeeze(c)
            centroid.append(np.mean(c, axis=0))
        # print(centroid)
        return centroid


def rotate_img(img, angle, center):
    img = tf.ToPILImage()(img)
    rot_img = tf.functional.rotate(img, angle, center=center)
    return np.array(rot_img)


def crop_img(img, center):
    row, col = center + 128
    # print(row, col)
    npz = np.zeros((4, 64, 64))
    for i in range(2, 9, 2):
        npz[i//2-1, ...] = cv2.resize(img[row-16*i:row+16*i, col-16*i:col+16*i], (64, 64), interpolation=cv2.INTER_AREA)
    return np.copy(npz)


def img2npz(img, coord, centroid, pad, png_name, npz_name):
    # print(img.shape)
    patch = []
    label = []
    id_ = []

    coord = (coord * img.shape[0]).astype(int)
    coord = coord.reshape((-1, 1, 2))

    img = np.pad(img, pad_width=pad)

    if centroid is None:
        for idx in range(len(coord)):
            print(coord[idx, 0, :], coord.shape)
            patch.append(crop_img(img, coord[idx, 0, :]))

            npz = crop_img(img, coord[idx, 0, :])
            # print(np.max(img), np.max(npz))
            cv2.imwrite(png_name + '_{}.png'.format(idx), npz[2, ...])

            label.append(0)
            id_.append(np.asarray([coord[idx, 0, 0], coord[idx, 0, 1], 512, 0]))
            pass
    else:
        dis = np.linalg.norm(coord - centroid, axis=-1)
        # print(dis.shape, dis[:5])
        dis = np.min(dis, axis=-1)
        # print(dis.shape, dis[:5])
        for idx in range(len(coord)):
            # print(idx, coord[idx], dis[idx])
            if dis[idx] < 24:
                # print(coord[:, 0, 0])
                # print(idx, coord[idx, 0, 1] + pad, coord[idx, 0, 0] + pad)
                for i in range(3):
                    rot_img = rotate_img(img,  30 * i, (coord[idx, 0, 1]+pad, coord[idx, 0, 0]+pad))    # check
                    npz = crop_img(rot_img, coord[idx, 0, :])
                    # print(np.max(img), np.max(npz))
                    cv2.imwrite(png_name + '_{}_{}.png'.format(idx, i), npz[2, ...])
                    patch.append(crop_img(img, coord[idx, 0, :]))
                    label.append(1)
                    id_.append(np.asarray([coord[idx, 0, 0], coord[idx, 0, 1], dis[idx], i]))
            elif dis[idx] > 64:
                patch.append(crop_img(img, coord[idx, 0, :]))
                npz = crop_img(img, coord[idx, 0, :])
                # print(np.max(img), np.max(npz))
                cv2.imwrite(png_name + '_{}.png'.format(idx), npz[2, ...])
                label.append(0)
                id_.append(np.asarray([coord[idx, 0, 0], coord[idx, 0, 1], dis[idx], 0]))
            # include in intermediate range patches
            else:
                patch.append(crop_img(img, coord[idx, 0, :]))
                npz = crop_img(img, coord[idx, 0, :])
                cv2.imwrite(png_name + '_{}.png'.format(idx), npz[2, ...])
                label.append(-1)
                id_.append(np.asarray([coord[idx, 0, 0], coord[idx, 0, 1], dis[idx], 0]))

    patch_npz = np.stack(patch)
    print(patch[0].shape, patch_npz.shape)
    label_npz = np.stack(label).reshape(-1, 1)
    print(label_npz.shape, np.min(label_npz), np.max(label_npz))
    id_npz = np.stack(id_)
    print(id_npz.shape)
    np.savez_compressed(npz_name + '.npz', img=patch_npz, label=label_npz, id=id_npz)
    print(npz_name)
    return


def main(ip_dir, op_npz_dir, op_png_dir, total_pts, pad):
    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in file_list:

        if not os.path.exists(ip_dir+file):
            continue

        if file.startswith('.') or 'mask' not in file:
            continue

        id_ = '_'.join(file.split('_')[:3])    # for the other two
        print(file, id_)

        op_npz_name = op_npz_dir + id_ + '.npz'
        op_png_name = op_png_dir + id_ + '_curved.png'

        mask_file_name = ip_dir + id_ + '_mask.png'
        if not os.path.exists(mask_file_name):
            continue

        # print(mask_file_name)
        mask = cv2.imread(mask_file_name, 0)
        # print(mask.shape, np.max(mask), np.min(mask))

        center_line_mask = np.copy(mask)
        center_line_mask[center_line_mask >= 20] = 255
        center_line_mask[center_line_mask < 20] = 0

        # resize may not be necessary
        center_line_mask = cv2.resize(center_line_mask, (128, 128), interpolation=cv2.INTER_AREA)
        coord, _ = mask2curve(center_line_mask, op_png_name, total_pts=total_pts)
        # print(coord)

        stenosis = center_stenosis(mask)

        # check stenosis labelling, done
        # if len(centroid) > 1:
        #     print('!!!', ip_dir + file, ip_dir, file)
        #     print(centroid)

        img = cv2.imread('{}{}.png'.format(ip_dir, id_), 0)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        print(file, stenosis, coord)

        if stenosis is not None:
            stenosis_img = draw_sten(np.copy(bgr_img), stenosis)
            # print(stenosis)
            stenosis = np.vstack(stenosis).reshape((1, -1, 2))
            stenosis[:, :, [0, 1]] = stenosis[:, :, [1, 0]]
            # print(stenosis, stenosis.shape)
        else:
            stenosis_img = np.copy(bgr_img)

        img2npz(img, coord, stenosis, pad, op_png_dir + id_, op_npz_dir + id_)

        # center_line_img = draw_pts(np.copy(bgr_img), coord, n_pts=total_pts)
        # combine_img = np.concatenate((stenosis_img, center_line_img), axis=1)
        # cv2.imwrite(op_png_name, combine_img)


if __name__ == '__main__':
    ip_dir = "/home/chentyt/Documents/4tb/Annotation/v2_combined_annotate_20-10-2021/"
    op_npz_dir = "/home/chentyt/Documents/4tb/Tiana/P100/Data/RCA_annotated_v2/patch_npz/"
    op_png_dir = "/home/chentyt/Documents/4tb/Tiana/P100/Data/RCA_annotated_v2/patch_png/"
    main(ip_dir, op_npz_dir, op_png_dir, total_pts=256, pad=128)