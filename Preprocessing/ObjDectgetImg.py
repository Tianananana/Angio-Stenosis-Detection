import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur



# modified from ImageProcessor
def get_bbox(mask):
    """
    Returns array of shape (n, 4) [xmin, ymin, xmax, ymax] for n bboxes
    :param mask: array mask of shape H * W
    """
    filter_mask = (mask == 22).astype('uint8')
    pts = cv2.connectedComponentsWithStats(filter_mask, connectivity=8)
    stats = pts[2]
    # print('stats', stats)
    stats = stats[np.argsort(stats[:, -1])]
    stats = stats[:-1, :]  # remove largest (background) bounding box
    ymin, xmin = stats[:, 0], stats[:, 1]
    ymax, xmax = ymin + stats[:, 2], xmin + stats[:, 3]
    bbox = np.vstack([xmin, ymin, xmax, ymax]).T
    return bbox


def draw_sten(img, coord):
    """
    :param img: tensor/numpy array
    :param coord: tensor of shape (no_pts, 2), to plot
    """
    if type(img) == torch.Tensor:  # add checking step
        img = img.numpy().astype('i')
    for c in coord:  # image processed in numpy
        row = int(c[0])
        col = int(c[1])
        img[row - 3:row + 4, col, :] = [0, 0, 255]
        img[row, col - 3:col + 4, :] = [0, 0, 255]
    return img


def mask_img(coord, img, id_, op_dir, cutoff=50, k=25, s=512 // 2 + 1, alpha=0.02, plot_checkpt=False):
    """
    Returns masked image based on coordinates of centreline
    :param coord: tensor of shape (n_pts, 2)
    :param img: tensor/numpy array
    :param id_:
    :param op_dir:
    :param cutoff: threshold of distance from coord for mask
    :param k, s: kernel, sigma for smoothing
    :param alpha: tuning param for exponential function
    :plot_checkpt: plot the intermediate images. Default: False
    """
    # initialise coords
    all_coord = torch.zeros((512 * 512, 2))

    # TODO torch arrange + reshape to get y
    for i in range(512):
        all_coord[i * 512:i * 512 + 512, 0] = i
        all_coord[i * 512:i * 512 + 512, 1] = torch.arange(512)
    all_coord = torch.broadcast_to(all_coord, (256, 512 * 512, 2))

    # get distance mask wrt coordinates
    dist = torch.cdist(all_coord, torch.unsqueeze(coord, 1))
    min_dist = torch.min(dist, 0).values
    distance_mask = torch.reshape(min_dist, (512, 512))
    distance_mask = torch.clamp(distance_mask, min=0, max=255)
    # distance_mask = (distance_mask > cutoff) * distance_mask
    distance_mask[distance_mask < cutoff] = 0

    # Smooth mask
    gaussian = GaussianBlur(kernel_size=k, sigma=s)
    distance_mask = gaussian(torch.unsqueeze(distance_mask, 0))
    distance_mask = torch.squeeze(distance_mask)

    # weight function
    factor = 1 / torch.exp(alpha * distance_mask)

    # smoothing the mask
    # factor = gaussian(torch.unsqueeze(factor, 0))
    # factor = torch.squeeze(factor)

    combined_img = img * factor
    combined_img = torch.clamp(combined_img, min=0, max=255)

    if plot_checkpt:
        # plot_img([img, distance_mask, factor, combined_img], id_, op_dir + 'png2/',
        #          ['Image', 'Distance', 'Exp', 'Final'])
        plot_img([img, factor, combined_img], id_, op_dir + 'png2/',
                 ['Original', 'Mask', 'Final'])


    return combined_img


def plot_bbox(bbox_coord, image, color=[255, 0, 0], overlap=False):
    """
    Returns plotted bbox images
    :param bbox_coord: array of shape (n, 4) [xmin, ymin, xmax, ymax] for n bboxes
    :param image: original image
    """
    img = np.copy(image)
    if img.shape[0] == 1:   # handle wrong shape format
        img = np.array(np.broadcast_to(np.moveaxis(img, 0, -1), (512, 512, 3)))

    if len(img.shape) == 2:     # handle 1 channel image
        img = np.array(np.broadcast_to(np.expand_dims(img, -1), (512, 512, 3)))

    for xmin, ymin, xmax, ymax in bbox_coord:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        if overlap:
            img[xmin: xmax, ymin - 3: ymin, 0] = 255
            img[xmin: xmax, ymax - 1: ymax + 2, 0] = 255
            img[xmin - 3: xmin, ymin: ymax, 0] = 255
            img[xmax - 1: xmax + 2, ymin: ymax, 0] = 255

        else:
            img[xmin: xmax, ymin - 3: ymin, :] = color
            img[xmin: xmax, ymax - 1: ymax + 2, :] = color
            img[xmin - 3: xmin, ymin: ymax, :] = color
            img[xmax - 1: xmax + 2, ymin: ymax, :] = color

    return img.astype(np.uint8)


def plot_img(imgs, id_, op_dir, *labels):
    """
    :param imgs: list of images to view
    :param id_: frame id
    :praram op_dir: output directory
    :param labels: list of labels. optional
    """
    n = len(imgs)
    f, ax = plt.subplots(1, n)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    if labels:
        labels = labels[0]
        for i in range(n):
            ax[i].set_title(labels[i])
    # plt.show()
    plt.savefig(op_dir + id_ + '.png')


def main():
    from ImageProcesser import mask2curve
    # split id into img and mask
    files = os.listdir(ip_dir)
    id_list = [i.split('.')[0] for i in files]
    for id_ in id_list:
        if 'mask' in id_:
            continue

        print(id_ + '\t')
        img = cv2.imread(ip_dir + id_ + '.png', 0)
        mask = cv2.imread(ip_dir + id_ + '_mask.png', 0)

        # 1.  Get 4 bbox coords: [xmin, ymin, xmax, ymax]
        bbox = get_bbox(mask)
        print('bbox', bbox)

        # 2. Masking surrounding region by getting min fr norm dist
        mask[mask >= 20] = 255
        mask[mask < 20] = 0
        coord, new_mask = mask2curve(cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA), op_dir, 256)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        # mask = torch.from_numpy(mask).type(torch.FloatTensor)
        coord = torch.from_numpy(coord * 512).type(torch.FloatTensor)


        # if not os.path.exists(op_dir + 'png2/'):
        #     os.makedirs(op_dir + 'png2')
        # masked_img = mask_img(coord, img, id_, op_dir, alpha=alpha, plot_checkpt=plot_checkpt)

        # 3. Save as npz: org image, masked image, centreline coord, bounding boxes
        if save_npz:
            if not os.path.exists(op_dir + 'npz_org/'):
                os.makedirs(op_dir + 'npz_org')

            # np.savez_compressed(op_dir + f'npz_org/{id_}.npz', id=np.array(id_.split('_')), mask_overlap=masked_img, coord=coord, bbox=bbox)
            np.savez_compressed(op_dir + f'npz_org/{id_}.npz', id=np.array(id_.split('_')), img=img, coord=coord, bbox=bbox)

        if save_plot_bbox:
            if not os.path.exists(op_dir + 'bbox_img/'):
                os.makedirs(op_dir + 'bbox_img/')
            plt.imsave(op_dir + f'bbox_img/{id_}.png', plot_bbox(bbox, img).astype('uint8'))
        pass

    pass


if __name__ == '__main__':
    ip_dir = "/home/chentyt/Documents/4tb/Annotation/v3_combined_annotate_24-01-2021/"  # directory of org
    op_dir = "/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Data/"
    alpha = 0.06

    save_npz = False
    plot_checkpt = False  # save processing images
    save_plot_bbox = False # Save plotted bbox (for checking)
    main()
