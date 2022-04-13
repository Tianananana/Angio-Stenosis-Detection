"""
Reorganize npz files into train, test, valid
"""

import os
import sys
import pandas as pd
import numpy as np


def read_npz(ip_folder, list_, op_name):

    for i in range(len(list_)):
        print(list_[i])
        id_ = list_[i].split('/')[-1].split('.')[0].split('_')
        print(id_)
        patient = int(id_[0])
        video = int(id_[1])
        frame = int(id_[2])

        with np.load(ip_folder + list_[i]) as npz:
            print(npz['mask_overlap'].shape, npz['coord'].shape, npz['bbox'], npz['id_'])
            # print(npz['img'].shape, npz['coord'].shape, npz['bbox'], npz['id'])

            overlap = np.expand_dims(npz['mask_overlap'], axis=0)
            # overlap = np.expand_dims(npz['img'], axis=0)
            coord = np.expand_dims(npz['coord'], axis=0)
            bbox = npz['bbox']
            id_ = np.expand_dims(npz['id_'], axis=0)

            try:
                overlap_combine = np.concatenate((overlap_combine, overlap), axis=0)
                coord_combine = np.concatenate((coord_combine, coord), axis=0) #might not be needed in Faster RCNN
                id_combine = np.concatenate((id_combine, id_), axis=0)
            except NameError:
                overlap_combine = overlap
                coord_combine = coord
                id_combine = id_


            for xmin, ymin, xmax, ymax in bbox:
                row = np.array([patient, video, frame, xmin, ymin, xmax, ymax])
                row = np.expand_dims(row, axis=0)
                try:
                    bbox_combine = np.concatenate((bbox_combine, row), axis=0)
                except NameError:
                    bbox_combine = np.empty((0, 7))



    overlap_combine = overlap_combine.astype(np.uint8)
    print(overlap_combine.shape, coord_combine.shape, bbox_combine.shape, id_combine.shape)
    id_combine = id_combine.astype(int)
    n_boxes = bbox_combine.shape[0]
    bbox_combine = np.concatenate((bbox_combine, np.ones((n_boxes, 1))), axis=-1)
    bbox_combine = bbox_combine.astype(int)

    np.savez(op_name, id=id_combine, img=overlap_combine, bbox=bbox_combine, coord=coord_combine)
    return


def main():
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    csv = pd.read_csv(op_dir + 'IdOnlySplit.csv')

    columns = ('train', 'valid', 'test')
    df = pd.DataFrame(csv, columns=columns)

    train_id = [int(i) for i in list(df['train'].dropna())]
    valid_id = [int(i) for i in list(df['valid'].dropna())]
    test_id = [int(i) for i in list(df['test'].dropna())]

    file_list = os.listdir(ip_dir)
    file_list.sort()

    train = []
    valid = []
    test = []
    for file in file_list:
        v_name = int(file.split('_')[0])
        # print(file, v_name)
        # print(train_id[1] == v_name, type(v_name), type(train_id[1]), type(test_id[1]))
        if v_name in train_id:
            train.append(file)
        elif v_name in valid_id:
            valid.append(file)
        elif v_name in test_id:
            test.append(file)
        else:
            sys.exit('Unclassified file {}'.format(file))


    print('processing train...')
    read_npz(ip_dir, train, op_dir + f'train_{version}.npz')
    print('processing valid...')
    read_npz(ip_dir, valid, op_dir + f'valid_{version}.npz')
    print('processing test...')
    read_npz(ip_dir, test, op_dir + f'test_{version}.npz')


if __name__ == '__main__':
    ip_dir = '/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Data/npz_org/'
    op_dir = '/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Data/RCA_Split1/'
    split_id = 1
    version = 'ORG'
    main()
