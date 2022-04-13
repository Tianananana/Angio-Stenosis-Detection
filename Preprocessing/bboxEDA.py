"""
Exploratory Data Analysis of no. image and no. stenosis per image.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv(df_dir)
    train = np.array(df['train'])
    test = np.array(df['test'])
    valid = np.array(df['valid'])

    train = train[~np.isnan(train)].astype(np.int)
    test = test[~np.isnan(test)].astype(np.int)
    valid = valid[~np.isnan(valid)].astype(np.int)

    train_freq = []
    test_freq = []
    valid_freq = []

    for f in os.listdir(data_dir):
        npz = np.load(data_dir + f)
        boxes = npz['bbox']
        id_ = int(f.split('_')[0])
        count = boxes.shape[0] - 0.5

        if id_ in train:
            train_freq.append(count)

        if id_ in test:
            test_freq.append(count)

        if id_ in valid:
            valid_freq.append(count)


    plt.hist(np.array([train_freq, test_freq, valid_freq]), bins=np.arange(min(test_freq), max(test_freq) + 2, 1), histtype='bar', stacked=False,
             edgecolor='black', linewidth=1.2, color=['b', 'orange', 'red'], label=['train', 'test', 'valid'])
    plt.legend()
    plt.xticks([0, 1, 2, 3, 4])
    plt.title("Stenosis Distribution")
    plt.xlabel("Stenosis per image")
    plt.ylabel("Count")
    plt.show()
    if save:
        plt.savefig(f'{op_dir}/StenosisDistBar.png')


if __name__ == "__main__":
    data_dir = '/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Data/npz/'
    df_dir = "/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Data/RCA_Split1/IdOnlySplit.csv"
    op_dir = '/home/chentyt/Documents/4tb/Tiana/P100ObjDet/bbox_EDA'
    save = True
    main()
