import glob
import random
import torch
import os
import os.path as op
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv
import h5py

def _bytes2img(img):
    # img_np = np.frombuffer(img_bytes, np.uint8)
    # img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    # img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    return img


class ChallengeDataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/ChallengeDataset/',
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/ChallengeDataset/',
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        #with open(self.meta_info_path, 'r') as fin:
            #self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        #self.io_opts_dict['client_keys'] = ['lq', 'gt']

        self.label_list = []
        for i in range(1, 201):
            fi = str(i).zfill(3)
            hdfFile = h5py.File('/public/rawvideo/label1/' + fi + '_label.hdf5', 'r')
            dataset = hdfFile.get('PQF_label')
            a = dataset[:]

            self.label_list.append(a)
            hdfFile.close()
        self.nfs_list = []
        for i in range(200):
            clip = "{0:0=3d}".format(i+1)
            nfs = len(os.listdir('/public/rawvideo/GT/' + clip))
            self.nfs_list.append(nfs)

        name_file = os.listdir('/public/rawvideo/GT')
        #self.total_lr_list = []
        self.total_gt_list = []
        self.folder_len = []
        self.folder_num = []
        for item in name_file:
            folder_name = '/public/rawvideo/GT/' + item
            image_list = os.listdir(folder_name)
            num = []
            for yy in image_list:
                image_name = folder_name + '/' + yy
                # print(int(image_name[-7:-4]))
                num = np.append(num, int(image_name[-7:-4]))
                num = np.sort(num)
            for image_idx in range(len(num)):
                a = int(num[image_idx])
                a = "{0:0=3d}".format(a)
                #self.total_lr_list.append(
                    #'/public/rawvideo/Challenge/RGB/Train_data/LQ' + '/' + item + '/' + a + '.png')
                self.total_gt_list.append(
                    '/public/rawvideo/GT' + '/' + item + '/' + a + '.png')
            self.folder_num.append(int(item))
            self.folder_len.append(len(image_list))

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        #if radius == 0:
        #    self.neighbor_list = [4, 4, 4]  # always the im4.png
        #else:
        #    nfs = 2 * radius + 1
        #    self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        #if self.file_client is None:
            #self.file_client = FileClient(
                #self.io_opts_dict.pop('type'), **self.io_opts_dict
            #)
        # random reverse
        #if self.opts_dict['random_reverse'] and random.random() < 0.5:
        #    self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========
        #index是xxxx行
        #001/006.png
        yy = int(self.total_gt_list[index][-11:-8])#视频index 1
        #image_ord = int(self.total_gt_list[index][-7:-4])#图片index 6

        #z = self.folder_num.index(yy)#视频index001的 index 0
        #a = int(np.sum(self.folder_len[:z]))
        #c = int(self.total_gt_list[index][-7:-4])#图片index 6

        lr_names = []
        hr_names = self.total_gt_list[index]
        gt_index = int(self.total_gt_list[index][-7:-4])
        mid = gt_index
        l, r = mid, mid
        nfs = self.nfs_list[yy-1]
        if mid == 1:
            l = 1
        else:
            l = mid - 1
        if mid == nfs:
            r = nfs
        else:
            r = mid + 1
        video_num = yy
        label_list1 = self.label_list[int(video_num) - 1]
        while label_list1[l - 1] != 1:
            l -= 1
        while label_list1[r - 1] != 1:
            r += 1
        neighbor_list = [l, mid, r]
        neighbor_list = np.clip(neighbor_list, 1, nfs)
        for i in range(len(neighbor_list)):
            a = neighbor_list[i]
            a = "{0:0=3d}".format(a)
            lr_names.append('/public/rawvideo/LQ/'+self.total_gt_list[index][-11:-8]+'/'+a+'.png')
        '''
        if (int(image_ord) <= 3):
            # print('a')
            temp_lr_list = self.total_lr_list[a + c - image_ord:a + c + 3]
            num = 7 - len(self.total_lr_list[a + c - image_ord:a + c + 3])
            for i in range(num):
                lr_names.append(temp_lr_list[0])
            for item in temp_lr_list:
                lr_names.append(item)

        if (int(image_ord) > (self.folder_len[z] - 4)):
            # print('b')
            temp_lr_list = self.total_lr_list[a + c - 4:a + self.folder_len[z]]
            for item in temp_lr_list:
                lr_names.append(item)
            num = 7 - len(temp_lr_list)
            for i in range(num):
                lr_names.append(temp_lr_list[-1])

        if ((self.folder_len[z] - 4) >= int(image_ord)) and (int(image_ord) >= 4):
            # print('c')
            lr_names = self.total_lr_list[a + c - 4:a + c + 3]
        '''
        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        #key = self.keys[index]
        #img_gt_path = key[:-4]
        #clip, img_num = key.split('/')  # key example: 00001/0001/im1.png
        #nfs = self.nfs_list[int(clip)-1]
        #neighbor_list = [i for i in range(int(img_num[:-4])-3,int(img_num[:-4])+4)]
        #neighbor_list = np.clip(neighbor_list, 1, nfs)
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            lr_names.reverse()
        #print(hr_names)
        img_bytes = cv2.imread(hr_names)
        img_gt = _bytes2img(img_bytes)  # (H W 1)
        #print(img_gt_path)
        # get the neighboring LQ frames
        img_lqs = []
        #print(lr_names)

        for neighbor in lr_names:
            img_bytes =cv2.imread(neighbor)
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)



        # ==========
        # data augmentation
        # ==========

        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, hr_names
            )

        # flip, rotate

        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        #print(np.shape(img_gt))
        #print(np.shape(img_lqs))

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.total_gt_list)


class VideoTestChallengeDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/ChallengeDataset/',
            self.opts_dict['gt_path']
        )
        self.lq_root = op.join(
            'data/ChallengeDataset/',
            self.opts_dict['lq_path']
        )

        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [],
            'lq_indexes': [],
            'h': [],
            'w': [],
            'index_vid': [],
            'name_vid': [],
        }

        self.label_list = []
        for i in range(1, 21):
            fi = str(i).zfill(3)
            hdfFile = h5py.File('/public/rawvideo/test_hdf52/' + fi + '_label.hdf5', 'r')
            dataset = hdfFile.get('PQF_label')
            a = dataset[:]

            self.label_list.append(a)
            hdfFile.close()

        # gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))

        gt_path_list = sorted(os.listdir(op.join(self.gt_root)))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            # w, h = map(int, name_vid.split('_')[-2].split('x'))
            w, h = 960, 536
            # nfs = int(name_vid.split('.')[-2].split('_')[-1])
            nfs = len(os.listdir(op.join(self.gt_root, gt_vid_path)))
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
            )

            for iter_frm in range(nfs):


                mid = iter_frm

                l, r = mid, mid
                if mid == 0:
                    l = 0
                else:
                    l = mid - 1
                if mid == nfs - 1:
                    r = nfs - 1
                else:
                    r = mid + 1
                video_num = name_vid

                label_list1 = self.label_list[int(video_num) - 1]

                while label_list1[l] != 1:
                    l -= 1
                while label_list1[r] != 1:
                    r += 1
                lq_indexes = [l, mid, r]

                # lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        gt_img = op.join(self.gt_root + str(self.data_info['gt_path'][index]),
                         "{0:0=3d}".format((self.data_info['gt_index'][index] + 1)) + '.png')
        img = cv2.imread(gt_img)
        img_gt = np.array(
            np.squeeze(img)
        ).astype(np.float32) / 255.  # (H W 3)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            lq_img = op.join(str(self.data_info['lq_path'][index]), "{0:0=3d}".format(lq_index + 1) + '.png')
            img = cv2.imread(lq_img)
            img_lq = np.array(
                np.squeeze(img)
            ).astype(np.float32) / 255.  # (H W 3)
            img_lqs.append(img_lq)
        # no any augmentation

        # to tensor
        img_lqs.append(img_gt)
        # print(np.shape(img_lqs))
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        # print(np.shape(img_gt))

        return {
            'lq': img_lqs,  # (T 3 H W)
            'gt': img_gt,  # (3 H W)
            'name_vid': self.data_info['name_vid'][index],
            'index_vid': self.data_info['index_vid'][index],
        }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num
