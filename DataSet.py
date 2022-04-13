import numpy as np
import torch
import random
import math
import torchvision.transforms as transforms


def rotate_box(boxes, angle, o=256):
    """
    Args:
        boxes: list of bbox coords:[xmin, ymin, xmax, ymax]
        angle: angle in degrees
        o: index of centre
    Returns:
        Rotated bbox coords
    """
    angle = math.radians(angle)  # convert to radians
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        x1_rot = (x1 - o) * math.cos(angle) - (y1 - o) * math.sin(angle) + o
        y1_rot = (x1 - o) * math.sin(angle) + (y1 - o) * math.cos(angle) + o
        x2_rot = (x2 - o) * math.cos(angle) - (y2 - o) * math.sin(angle) + o
        y2_rot = (x2 - o) * math.sin(angle) + (y2 - o) * math.cos(angle) + o
        x1_new, x2_new = min(x1_rot, x2_rot), max(x1_rot, x2_rot)
        y1_new, y2_new = min(y1_rot, y2_rot), max(y1_rot, y2_rot)
        new_boxes.append(list([x1_new, y1_new, x2_new, y2_new]))
    return new_boxes


def hflip_box(boxes, o=256):
    """
    y axis
    """
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        y1_rot, y2_rot = y1 - 2 * (y1 - o), y2 - 2 * (y2 - o)
        y1_new, y2_new = min(y1_rot, y2_rot), max(y1_rot, y2_rot)
        new_boxes.append(list([x1, y1_new, x2, y2_new]))
    return new_boxes


def vflip_box(boxes, o=256):
    """
    x axis
    """
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        x1_rot, x2_rot = x1 - 2 * (x1 - o), x2 - 2 * (x2 - o)
        x1_new, x2_new = min(x1_rot, x2_rot), max(x1_rot, x2_rot)
        new_boxes.append(list([x1_new, y1, x2_new, y2]))
    return new_boxes


def augment(img, coord):
    """
    Args:
        Perform random rotation of image via 90 deg rotation and flipping
        img: (1, 512, 512)
        coord: list of coords
    Returns:
        Augmented image
    """
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    img = transforms.functional.rotate(img, angle)
    # if coord:
    coord = rotate_box(coord, angle, o=256)

    i, j = random.randrange(2), random.randrange(2)
    if i:
        img = transforms.functional.hflip(img)
        # if coord:
        coord = hflip_box(coord, o=256)
    if j:
        img = transforms.functional.vflip(img)
        # if coord:
        coord = vflip_box(coord, o=256)

    coord = torch.tensor(coord)

    if coord.size()[-1] != 4:
        coord = torch.empty([0, 4])
    return img, coord
    # return img


class DataSet(torch.utils.data.Dataset):
    def __init__(self, file_dir, npz, transform=True):
        npz = np.load(file_dir + npz + '.npz')
        self.id_ = torch.from_numpy(npz['id'])
        self.img = torch.tensor(npz['img'], dtype=torch.float32)  # masked image
        self.bbox = torch.from_numpy(npz['bbox'])
        self.transform = transform
        # print(f'Id: {self.id_} \t Overall img shape: {self.img.shape} \t bbox label shape: {self.bbox}')

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, i):
        patient = self.id_[i, 0]
        video = self.id_[i, 1]
        frame = self.id_[i, 2]
        img = self.img[i]
        img = torch.unsqueeze(img, 0)

        img = img.cuda()
        check = self.bbox[(self.bbox[:, 0] == patient) &
                          (self.bbox[:, 1] == video) &
                          (self.bbox[:, 2] == frame)]

        # generate dummy bboxes if image has no bounding box
        if check.shape == 0:
            coord, label = np.empty((0, 4)), np.empty((0, 1))
        else:
            coord, label = check[:, 3:7], check[:, 7]

        # add augmentation
        if self.transform:
            img, coord = augment(img, coord)

        coord, label = coord.cuda(), label.cuda()
        target = {'boxes': coord,
                  'labels': label}

        return img, target
