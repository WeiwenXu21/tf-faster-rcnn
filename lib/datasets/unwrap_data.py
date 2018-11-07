# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from scipy.io import loadmat
import numpy as np

#train = loadmat('../../data/wider_face/wider_face_split/wider_face_train.mat')
train = loadmat('../../data/wider_face/wider_face_split/wider_face_val.mat')
#test = loadmat('../../data/wider_face/wider_face_split/wider_face_test.mat')

#train_bbox = train['face_bbx_list']
#train_file = train['file_list']
#train_vlabel = train['invalid_label_list']

#val_bbox

def unwrap_data(data):
    data_reshaped = []
    for item in data:
        tmp = item[0]
        tmp_file = []
        for file in tmp:
            tmp_file.append(file[0][0])
        tmp_file = np.array(tmp_file)
        data_reshaped.append(tmp_file)
    data_reshaped = np.array(data_reshaped)
    return data_reshaped

def unwrap_bbox_data(data):
    data_reshaped = []
    for item in data:
        tmp = item[0]
        tmp_file = []
        for file in tmp:
            tmp_file.append(file[0])
        tmp_file = np.array(tmp_file)
        data_reshaped.append(tmp_file)
    data_reshaped = np.array(data_reshaped)
    return data_reshaped

def setup_data(data):
    bbox = data['face_bbx_list']
    file = data['file_list']
    vlabel = data['invalid_label_list']

    unwrap_bbox = unwrap_bbox_data(bbox)
    unwrap_file = unwrap_data(file)
    unwrap_vlabel = unwrap_bbox_data(vlabel)

#    events, img_index = np.shape(unwrap_bbox)

    print(unwrap_bbox[0][2].shape)
    print(unwrap_file[0][2])
    data_dict = {}
    for i in range(len(unwrap_file)):
        for j in range(len(unwrap_file[i])):
            b = []
            for l in range(len(unwrap_vlabel[i][j])):
                if not unwrap_vlabel[i][j][l] ==1:
                    b.append(unwrap_bbox[i][j][l])
            b = np.array(b)
            data_dict[unwrap_file[i][j]] = b
    return data_dict


data_dict = setup_data(train)
np.save('../../data/wider_face/wider_face_split/val_bbox_dict.npy', data_dict)
#print(data_dict['0_Parade_marchingband_1_799'])












