import os
import time
import yaml
import argparse
import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torchvision
import utils  # my tool box
import dataset
from net_stdf import MFVQE


def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = opts_dict['train']['checkpointfile']

    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion']['unit']

    # ==========
    # open logger
    # ==========

    log_fp = open(opts_dict['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Test {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict['test'])}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create test data prefetchers
    # ==========
    
    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in dataset.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(dataset, test_ds_type)
    test_ds = test_ds_cls(
        opts_dict=opts_dict['dataset']['test'], 
        radius=radius
        )

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = utils.create_dataloader(
        dataset=test_ds, 
        opts_dict=opts_dict, 
        sampler=test_sampler, 
        phase='val'
        )
    assert test_loader is not None

    # create dataloader prefetchers
    test_prefetcher = utils.CPUPrefetcher(test_loader)

    # ==========
    # create & load model
    # ==========

    model = MFVQE(opts_dict=opts_dict['network'])

    checkpoint_save_path = opts_dict['test']['checkpoint_save_path']
    restore_address = opts_dict['test']['restore_address']
    if not os.path.exists(restore_address):
        os.mkdir(restore_address)
    msg = f'loading model {checkpoint_save_path}...'
    print(msg)
    log_fp.write(msg + '\n')

    checkpoint = torch.load(checkpoint_save_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    
    msg = f'> model {checkpoint_save_path} loaded.'
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    model.eval()

    # ==========
    # define criterion
    # ==========

    # define criterion
    assert opts_dict['test']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()

    # ==========
    # validation
    # ==========
                
    # create timer
    #total_timer = utils.Timer()

    # create counters
    per_aver_dict = dict()
    ori_aver_dict = dict()
    name_vid_dict = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict[index_vid] = utils.Counter()
        ori_aver_dict[index_vid] = utils.Counter()
        name_vid_dict[index_vid] = ""

    pbar = tqdm(
        total=test_num, 
        ncols=opts_dict['test']['pbar_len']
        )

    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    with torch.no_grad():
        name_vid_last = 0
        #print(val_data)
        while val_data is not None:
            # get data
            # gt_data = val_data['gt'].cuda()  # (B [RGB] H W)
            lq_data = val_data['lq'].cuda()  # (B T [RGB] H W)
            # index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            folder_address = os.path.join(restore_address, name_vid[:3])
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            if name_vid != name_vid_last:
                name_vid_last = name_vid
                frame_count = 0

            frame_count = frame_count+1
            frame_count_address = "{0:0=3d}".format(frame_count)

            if (frame_count % 10) == 0:
                #b, _, c, _, _  = lq_data.shape
                #assert b == 1, "Not supported!"

                #input_data = torch.cat(
                    #[lq_data[:,:,i,...] for i in range(c)],
                    #dim=1
                    #)  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W

                enhanced_data = model(lq_data)  # (B [RGB] H W)
                enhanced_data = model(lq_data)  # (B [RGB] H W)
                # enhanced_data = enhanced_data[:, [2, 1, 0], :, :]
                image_address = os.path.join(folder_address, frame_count_address)
                torchvision.utils.save_image(enhanced_data, image_address + ".png")


                restore_address1 = '/home/shik9/Public/1/STDF-PyTorch-2 (copy)/out1/'
                if not os.path.exists(restore_address1):
                    os.mkdir(restore_address1)
                folder_address1 = os.path.join(restore_address1, name_vid[:3])
                if not os.path.exists(folder_address1):
                    os.mkdir(folder_address1)
                if(frame_count % 10) == 0:
                    image_address1 = os.path.join(folder_address1, frame_count_address)
                    torchvision.utils.save_image(enhanced_data, image_address1 + ".png")

            # display
            pbar.set_description(
                "{:s}"
                .format(name_vid)
                )
            pbar.update()

            # fetch next batch
            val_data = test_prefetcher.next()

    # end of val
    pbar.close()

if __name__ == '__main__':
    main()