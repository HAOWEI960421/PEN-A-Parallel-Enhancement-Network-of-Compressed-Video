import os
import imquality.brisque as brisque
import PIL.Image

import numpy as np
import h5py
'''
for i in range(1,11):
    fi = str(i).zfill(3)
    os.makedirs('/public/rawvideo/test_png1/'+fi)
    os.system('ffmpeg -i /public/rawvideo/test_png1/'+ fi+'.mkv /public/rawvideo/test_png1/'+fi+'/%3d.png')
'''

path0 = '/public/rawvideo/'
foder_list = os.listdir(path0+'test_png')

for foder in foder_list:

    #isExists = os.path.exists(label_path)
    #if not isExists:
        #os.makedirs(label_path)

    path1 = path0 + 'test_png/'+foder
    fra_num = len(os.listdir(path1))
    list_psnr = []
    txt = path0+'test_hdf52/'+foder+'_psnr.txt'

    for i in range(1,fra_num+1):
        j = str(i).rjust(3, '0')
        path2 = path1+'/'+j+'.png'

        img = PIL.Image.open(path2)
        psnr =100- brisque.score(img)

        list_psnr.append(psnr)
        with open(txt, "a") as f:
            f.write("score:" + str(psnr) + "\n")

    print('/////////////////////////////////////////////////////////////////////')




    SSIM_ratio_thr = 0.75
    SSIM_thr = 0.6
    radius = 2
    list_index_ClipStart = [0]

    PQF_label = np.zeros((len(list_psnr),), dtype=np.int32)

    num_clip = len(list_index_ClipStart)

    for ite_ClipStart in range(num_clip):

        # extract PSNR clip
        index_start = list_index_ClipStart[ite_ClipStart]
        if ite_ClipStart < num_clip - 1:
            index_NextStart = list_index_ClipStart[ite_ClipStart + 1]
        else:
            index_NextStart = len(list_psnr)
        PSNR_clip = list_psnr[index_start: index_NextStart]

        # the first and the last frame must be PQF
        PQFLabel_clip = np.zeros((len(PSNR_clip),), dtype=np.int32)


        PQFLabel_clip[0] = 1
        PQFLabel_clip[-1] = 1

        # Make label
        for ite_frame in range(1, len(PQFLabel_clip) - 1):
            if (PSNR_clip[ite_frame] > PSNR_clip[ite_frame - 1]) and (PSNR_clip[ite_frame] >= PSNR_clip[ite_frame + 1]):
                PQFLabel_clip[ite_frame] = 1

        PQF_label[index_start: index_NextStart] = PQFLabel_clip

    #'make label and modify'
    dis_max = 3
    while True:

        PQF_index = np.where(PQF_label == 1)[0]
        PQF_distances = PQF_index[1:] - PQF_index[0:-1]
        TooLongDistance_order_list = np.where(PQF_distances > dis_max)[0]

        if len(TooLongDistance_order_list) == 0:  # None
            break
        # reason: monotony
        for ite_order in range(len(TooLongDistance_order_list)):

            TooLongDistance_order = TooLongDistance_order_list[ite_order]
            PQF_index_left = PQF_index[TooLongDistance_order]
            PQF_index_right = PQF_index[TooLongDistance_order + 1]

            if list_psnr[PQF_index_left] <= list_psnr[PQF_index_right]:
                PQF_label[PQF_index_right - 2] = 1
            else:
                PQF_label[PQF_index_left + 2] = 1

    save_path = path0+'test_hdf52/'+ foder +'_label.hdf5'
    f = h5py.File(save_path, "w")
    f.create_dataset('PQF_label', data=PQF_label)
    f.close()
#hdfFile = h5py.File('D:/download/make_label/label/001_label.hdf5', 'r')
#dataset1 = hdfFile.get('PQF_label')
#a = dataset1[:]
#print(dataset1)
