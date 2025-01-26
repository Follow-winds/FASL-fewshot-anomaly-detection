import os
import cv2
import json
import torch
import time
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import open_clip
from few_shot import memory, memory_win, memory_imgidx, memory_win_imgidx
from model import LinearLayer
from dataset import VisaDataset, MVTecDataset
from prompt_ensemble import encode_text_with_prompt_ensemble

from FASLModel import FASL, winSplit, countRes

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,4'

outlog = False

def sprint(strs):
    if outlog:
        print(strs)

def calWinAnoMap(winmap, masklist, winsize, picsize, device, usewinimgencoder, convRes, winzsize):
    '''
        winmap: (b*winnum) * n-1 * 2
        masklist: winnum * 4
        winsize: int 5
        winzsize: int 7
        picsize: int 37
        usewinimgencoder 是否使用了滑动窗口图像编码器
    '''    
        
    winnum = len(masklist)
    print('winnum: {}'.format(winnum))
    B, N, C = winmap.shape
    print('winmap shape: {}'.format(winmap.shape))
    if winzsize != winsize:                         #
        winmap = winmap.permute(0, 2, 1)
        H = int(np.sqrt(winmap.shape[2]))
        winmap = winmap.view(B, C, H, H)
        winmap = F.interpolate(winmap, size=winsize, mode='bilinear', align_corners=True)
        winmap = winmap.view(B, C, -1)
        winmap = winmap.permute(0, 2, 1)
        
    #
    if usewinimgencoder == 0 and winsize != picsize:
        winmap = winmap.permute(0, 2, 1)
        H = int(np.sqrt(N))
        winmap = winmap.view(B, C, H, H)
        # print('winmap shape: {}'.format(winmap.shape))
        winmap = F.interpolate(winmap, size=winsize, mode='bilinear', align_corners=True)
        # print('winmap shape: {}'.format(winmap.shape))
        winmap = winmap.view(B, C, -1)
        winmap = winmap.permute(0, 2, 1)
        # print('winmap shape: {}'.format(winmap.shape))
    B, N ,C = winmap.shape
        
    winmap = winmap.view(-1, winnum, N, C)                          # b * winnum * n-1 * 2
    b, _, _, _ = winmap.shape
    
    anoMaps = []
    for wm in winmap:                                               # wm: winnum * n-1 * 2
        anowm = wm[:, :, 1]                                         # winnum * n-1
        anowm = anowm.view(winnum, winsize, winsize)                # winnum * wsize * wsize
        
        anoMap = torch.zeros(picsize, picsize).to(device)
        calnum = torch.zeros(picsize, picsize).to(device)
        for i in range(winnum):
            mask = masklist[i]
            row_start = mask[0]
            row_end = mask[1]
            col_start = mask[2]
            col_end = mask[3]
            anoMap[row_start:row_end, col_start:col_end] += anowm[i]
            calnum[row_start:row_end, col_start:col_end] += 1
        # print(calnum)
        anoMap = anoMap / calnum
        anoMaps.append(anoMap)                                      # b * picsize * picsize
    anoMaps = torch.stack(anoMaps)
    anoMaps = anoMaps.view(b, -1)                                   # b * n    
    t = [1. - anoMaps, anoMaps]                                     
    t = torch.stack(t)                                              # 2 * b * n
    t = t.permute(1, 2, 0)                                          # b * n * 2
    if convRes != 0:
        t = convRes(t)                                                  # b * n * 2
    anoMaps = t[:, :, 1]
    anoMaps = anoMaps.view(b, picsize, picsize)
    return anoMaps
        
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs == []:
        return 0.0
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    calimg_size = args.calImage_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')
    
    # count res
    if args.count_res:
        rrcount = countRes(args.rrcount_path)
        # rrcount.to(device)
        coscount = countRes(args.coscount_path)
        # coscount.to(device)

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)    
    print("processer: {}".format(preprocess))
    
    model.eval().to(device)
    # tokenizer = open_clip.get_tokenizer(args.model)
    if args.winmode != 'nan':
        if args.usewinimgencoder:
            model_win, _, _ = open_clip.create_model_and_transforms(args.model, args.winZipSize*args.patchsize, pretrained=args.pretrained)
            model_win.eval().to(device)
        else:
            model_win = model
        
        # win spliter
        winSpliter = winSplit(winsize=args.winsize, stride=args.winstride, winZipSize=args.winZipSize)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
        

    # selfModel
    RR = FASL(
        useZeroCls = args.useZeroCls,
        useAdapter = args.useAdapter, 
        useAbs = args.useAbs, 
        useFirstBatchNorm = args.useFirstBatchNorm, 
        useConvRes = args.useConvRes,
        useSENet = args.useSENet,
        ResFusionMode = args.ResFusionMode,
        mode = args.RRmode
    )
    
    if args.winmode != 'nan':
        winRR = FASL(
            useZeroCls = False,
            useAdapter = args.useAdapter, 
            useAbs = args.useAbs, 
            useFirstBatchNorm = args.useFirstBatchNorm, 
            useConvRes = args.winuseConvRes,
            useSENet = args.useSENet,
            ResFusionMode = 'nan',
            mode = 'fewshot'
        )
    checkpoint = torch.load(args.checkpoint_path)
    RR.load_state_dict(checkpoint["trainable_RR"], strict=False)            # strict=False
    RR = RR.eval().to(device)
    
    if args.winmode != 'nan':
        winRR.load_state_dict(checkpoint["trainable_RR"], strict=False)
        winRR = winRR.eval().to(device)
        if args.useConvRes:
            convResModule = winRR.getConvRes()
        else:
            convResModule = 0

    # dataset
    transform = transforms.Compose([
            transforms.Resize((calimg_size, calimg_size)),
            transforms.CenterCrop(calimg_size),
            transforms.ToTensor()
        ])
    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test')
    else:
        test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names()

    # few shot memory
    if args.mode == 'few_shot':
        if False :  # args.useImgIdx
            mem_imgfeatures, mem_features = memory_imgidx(args.model, model, obj_list, dataset_dir, save_path, preprocess, transform,
                              args.k_shot, few_shot_features, dataset_name, device, args.useNorm)
        else:
            mem_features = memory(args.model, model, obj_list, dataset_dir, save_path, preprocess, transform,
                              args.k_shot, few_shot_features, dataset_name, device, args.useNorm)
        
        # win few shot
        if args.winmode != 'nan':
            if args.usewinmem:
                if args.useImgIdx:
                    win_mem_imgfeatures, win_mem_features = memory_win_imgidx(args.model, model_win, obj_list, dataset_dir, save_path, preprocess, transform,
                                args.k_shot, few_shot_features, dataset_name, device, winzsize=args.winZipSize, 
                                win_size=args.winsize, win_stride=args.winstride, usewinimgencoder=args.usewinimgencoder,
                                image_size=args.image_size, useNorm=args.useNorm)
                else:
                    win_mem_features = memory_win(args.model, model_win, obj_list, dataset_dir, save_path, preprocess, transform,
                                args.k_shot, few_shot_features, dataset_name, device, winzsize=args.winZipSize, 
                                win_size=args.winsize, win_stride=args.winstride, usewinimgencoder=args.usewinimgencoder,
                                image_size=args.image_size, useNorm=args.useNorm)
            else:
                win_mem_features = mem_features

    # text prompt
    # with torch.cuda.amp.autocast(), torch.no_grad():
    #     text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    results['img_path'] = []
    idxx = 1
    
    avg_use_time_micro = 0
    for items in test_dataloader:
        
        print(" {} / {} ----------------------------------------------------------------------\n ".format(idxx, len(test_data)))
        idxx += 1
        start_time = time.perf_counter()
        
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())
        
        sprint("input model img size: {}".format(image.shape))                                       # b * 3 * imgsize * imgsize
        sprint("input model lab size: {}".format(gt_mask.shape))                                     # b * 1 * calimgsize * calimgsize

        if args.winmode != 'nan':
            win_image = image.clone()
            # win_gt = gt_mask.clone()
            
            with torch.no_grad():
                win_images, win_masklist = winSpliter(win_image)                # (winnum * b) * 3 * winszie * winszie
                # win_gts, _ = winSpliter(win_gt)                                 # (winnum * b) * 1 * winszie * winszie
                
                if args.usewinimgencoder == 0:
                    win_images = F.interpolate(win_images, size=args.image_size, mode='bilinear', align_corners=True)   # (winnum * b) * 3 * winszie(518) * winszie
                


        with torch.no_grad(), torch.cuda.amp.autocast():

            # few shot
            if args.mode == 'few_shot':
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                if args.useNorm:
                    patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in patch_tokens]
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                if False:   # args.useImgIdx:
                    mem_imgfeature = mem_imgfeatures[cls_name[0]]
                    print("[memory] few shots img features size: {} bytes".format(mem_imgfeature.element_size() * mem_imgfeature.nelement()))
                    mimg = image_features @ mem_imgfeature.T       # 1 * k
                    maxImgIdx = torch.argmax(mimg, dim=-1)
                
                anomaly_maps_few_shot = []
                few_features = []
                cos_res = []
                for idx, p in enumerate(patch_tokens):                                                          # layer    # p: b*n*c
                    if 'ViT' in args.model:
                        p_ = p[0, 1:, :]                                                                        # n-1 * c
                    else:
                        p_ = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                    
                    if False:   # args.useImgIdx:
                        mem_feature = mem_features[cls_name[0]]                             # k * l * n-1 * c                        
                        mem_feature = mem_feature[maxImgIdx]                                # l * n-1 * c
                    else:
                        mem_feature = mem_features[cls_name[0]]
                    sprint("mem_feature size: {} * {}".format(len(mem_feature), mem_feature[0].shape))
                    mem_feature_l = mem_feature[idx]
                    sprint("mem_feature_l size: {}".format(mem_feature_l.shape))
                    print("[memory] few shots features size: {} bytes".format((mem_feature_l.element_size() * mem_feature_l.nelement()) * 4))
                    
                    if args.alignMode == 'mulmax':
                        m = p_ @ mem_feature_l.T
                        m_idx = torch.argmax(m, dim=1)                                      # n - 1
                        
                    if args.alignMode == 'diff2':
                        m_idx = []
                        for ft in p_:
                            ft = mem_feature_l - ft
                            ft = ft.norm(dim=1)
                            _m_idx = torch.argmin(ft)
                            m_idx.append(_m_idx)
                        m_idx = torch.stack(m_idx)                                          # n - 1
                            
                            
                    
                    #
                    if args.count_res:
                        cosval_l = pairwise.cosine_similarity(p_.cpu(), mem_feature_l.cpu())
                        cosval_l = (cosval_l + 1) / 2
                        # print(cosval_l.shape)
                        # print(cosval_l.max())
                        # print(cosval_l.min())
                        cosval_l = np.min((1. - cosval_l), 1)
                        cosval_l = torch.tensor(cosval_l)
                        sprint(cosval_l.max())
                        sprint(cosval_l.min())
                        # print(cosval_l.shape)
                        H = int(np.sqrt(len(cosval_l)))
                        cosval_l = F.interpolate(cosval_l.unsqueeze(0).unsqueeze(0).permute(0, 2, 1).view(1, 1, H, H),
                                                size=calimg_size, mode='bilinear', align_corners=True)          # 1 * 1 * 240 * 240
                        cos_res.append(cosval_l)
                    
                    
                    few_features_l = []
                    for ii in m_idx:
                        few_features_l.append(mem_feature_l[ii])
                    few_features_l = torch.stack(few_features_l)                            # n-1 * c
                    sprint("few_features_l size: {}".format(few_features_l.shape))
                    few_features.append(few_features_l)                                     # layer * n-1 * c
                few_features = torch.stack(few_features)
                few_features = few_features.unsqueeze(1)                                    # layer * 1 * n-1 * c
                sprint("few_features size: {} * {}".format(len(few_features), few_features[0].shape))
                few_features = few_features.to(device)
                
                
                if args.count_res:
                    cos_res = torch.stack(cos_res)                                          # l * 1 * 1 * 240 * 240
                    cos_res = cos_res.view(-1, calimg_size, calimg_size)                    # l * 240 * 240
                    la, _, _ = cos_res.shape
                    
                    cos_res = torch.sum(cos_res, dim=0) / la                                  # 240 * 240
                    cos_res = cos_res.unsqueeze(0)                                          # 1 * 240 * 240
                    
                
                patch_tokens = torch.stack(patch_tokens)                                    # layernum * b * n * c(1024)
                
                #
                res, fusion_res = RR(patch_tokens, few_features)                                               # l * b * n * 2
                
                anomaly_maps = []
                img_pr = 0                                                                          # img level
                s = 0
                    
                if args.RRmode == 'zeroshot':
                    args.ResFusionMode = 'nan'
                    
                if args.useZeroCls and args.ResFusionMode != 'nan':
                    res = fusion_res
                                   
                    
                for l in range(len(res)):                                                           # layernum
                    if args.useConvRes and l % 2  == 0 and not args.useZeroCls:
                        continue
                    if args.useConvRes and l % 2  == 0 and args.RRmode == 'zeroshot':
                        continue
                    s += 1
                    anomaly_map = res[l]                                                            # b(1) * n * 2
                    img_pr += anomaly_map[:,:,1].max()
                    B, N, C = anomaly_map.shape
                    H = int(np.sqrt(N))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=calimg_size, mode='bilinear', align_corners=True)
                    anomaly_maps.append(anomaly_map[:,1,:,:].cpu().numpy())                                                # l * b * 1 * imgsize * imgsize
                
                anomaly_map = np.sum(anomaly_maps, axis=0) / s
                img_pr = img_pr / s
                
                if args.winmode == 'nan':
                    results['pr_sp'].append(img_pr.cpu().item())
                
                #     cos = pairwise.cosine_similarity(mem_features[cls_name[0]][idx].cpu(), p.cpu())
                #     height = int(np.sqrt(cos.shape[1]))
                #     anomaly_map_few_shot = np.min((1 - cos), 0).reshape(1, 1, height, height)
                #     anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                #                                          size=calimg_size, mode='bilinear', align_corners=True)
                #     anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                # anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                # anomaly_map = anomaly_map + anomaly_map_few_shot
            
                    print("anomaly map size: {}".format(anomaly_map.shape))                                             # b * calimgsize * calimgsize
                    results['anomaly_maps'].append(anomaly_map)
                
                # winRR
                if args.winmode != 'nan':
                    if args.usewinimgencoder:
                        win_image_features, win_patch_tokens = model_win.encode_image(win_images, few_shot_features)
                        if args.useImgIdx:
                            win_image_features = F.normalize(win_image_features, p=2, dim=-1)                           # winnum * c
                        if args.useNorm:
                            win_patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in win_patch_tokens]
                        # win_image_feature /= win_image_feature.norm(dim=-1, keepdim=True)
                    else:
                        win_patch_tokens = []                   # layernum * (b*winnum) * n * c
                        win_img_features = []
                        for wimage in win_images:                                                                       # wimage: 3 * winszie * winszie
                            win_image_feature, win_patch_token = model_win.encode_image(wimage.unsqueeze(0), few_shot_features)    # layernum * 1 * n * c
                            if args.useImgIdx:
                                win_image_feature = F.normalize(win_image_feature, p=2, dim=-1)                             
                                win_img_features.append(win_image_feature)                                                  # winnum * c
                            if args.useNorm:
                                win_patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in win_patch_tokens]
                            win_patch_token = torch.stack(win_patch_token)
                            # print('win_patch_token shape : {}'.format(win_patch_token.shape))
                            win_patch_token = win_patch_token.squeeze(1)                                                            # layernum * n * c
                            # win_image_feature /= win_image_feature.norm(dim=-1, keepdim=True)
                            win_patch_tokens.append(win_patch_token)
                        win_patch_tokens = torch.stack(win_patch_tokens)                # (b*winnum) * layernum * n * c
                        win_patch_tokens = win_patch_tokens.permute(1,0,2,3)            # layernum * (b*winnum) * n * c
                        if args.useImgIdx:
                            win_img_features = torch.stack(win_img_features)            # winnum * c
                        # print('win_patch_tokens shape : {}'.format(win_patch_tokens.shape))
                    
                    if args.useImgIdx:
                        print("[memory] win few shots img features size: {} bytes".format(win_mem_imgfeatures[cls_name[0]].element_size() * win_mem_imgfeatures[cls_name[0]].nelement()))
                        wmi = win_image_features @ win_mem_imgfeatures[cls_name[0]].T                # winnum * (winnum * k)
                        wmaxImgIdx = torch.argmax(wmi, dim=-1)                           # winnum
                    
                    few_features = []
                    for idx, p in enumerate(win_patch_tokens):                                                      # layer    # p: (b*winnum)*n*c
                        if 'ViT' in args.model:
                            p_ = p[:, 1:, :]                                                                        # (b*winnum) * n-1 * c
                        else:
                            p_ = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()
                        
                        if args.useImgIdx:
                            win_mem_feature = win_mem_features[cls_name[0]]                         # (winnum * k) * l * n-1 * c
                            win_mem_feature = win_mem_feature[wmaxImgIdx.cpu()]                            # winnum * l * n-1 * c
                            win_mem_feature = win_mem_feature.permute(1, 0, 2, 3)                           # l * winnum * n-1 * c
                            # print('****** size : {}'.format(win_mem_feature.shape))
                            L, W, N, C = win_mem_feature.shape
                            win_mem_feature = win_mem_feature.reshape(L, -1, C)                                # l * (winnum * n-1) * c
                        else:
                            win_mem_feature = win_mem_features[cls_name[0]]                                 #
                        
                        sprint("win mem_feature size: {} * {}".format(len(win_mem_feature), win_mem_feature[0].shape))   # l * n_ * c
                        win_mem_feature_l = win_mem_feature[idx]                                        #
                        win_mem_feature_l = win_mem_feature_l.half().to(device)
                        sprint("win mem_feature_l size: {}".format(win_mem_feature_l.shape))                             # n_ * c
                        print("[memory] win mem few shots features size: {} bytes".format((win_mem_feature_l.element_size() * win_mem_feature_l.nelement()) * 4))
                        
                        if args.alignMode == 'mulmax':                        
                            wm = p_ @ win_mem_feature_l.T                                                    # (b*winnum) * n-1 * n_
                            wm_idx = torch.argmax(wm, dim=2)                                                 # (b*winnum) * n-1
                        
                        if args.alignMode == 'diff2':
                            wm_idx = []
                            for bt in p_:
                                _idx = []
                                for ft in bt:
                                    ft = win_mem_feature_l - ft
                                    ft = ft.norm(dim=1)
                                    _m_idx = torch.argmin(ft)
                                    _idx.append(_m_idx)
                                _idx = torch.stack(_idx)                                           # n - 1
                                wm_idx.append(_idx)
                            wm_idx = torch.stack(wm_idx)                                          # (b * winnum) * n - 1
                        
                        win_few_feature = []
                        for bwi in wm_idx:                                                               # bwi: n-1                            
                            few_features_l = []
                            for ii in bwi:
                                few_features_l.append(win_mem_feature_l[ii])
                            few_features_l = torch.stack(few_features_l)                            # n-1 * c
                            # print("few_features_l size: {}".format(few_features_l.shape))
                            win_few_feature.append(few_features_l)                                  # (b*winnum) * n-1 * c
                        win_few_feature = torch.stack(win_few_feature)
                        few_features.append(win_few_feature)                                     # layer *(b*winnum)* n-1 * c
                    few_features = torch.stack(few_features)                                     # layer *(b*winnum)* n-1 * c
                    
                    few_features = few_features.to(device)
                
                    # win_patch_tokens = torch.stack(win_patch_tokens)                             # layernum * (b*winnum) * n * c(1024)
                
                    #
                    win_res, _ = winRR(win_patch_tokens, few_features)                                               # l * (b*winnum) * n-1 * 2
                    
                    win_anomaly_maps = []
                    win_img_pr = 0                                                                          # 图像级
                    s = 0    
                    
                    for l in range(len(win_res)):                                                           # layernum
                        if args.winuseConvRes and l % 2:
                            continue
                        
                        s += 1
                        winmaps = win_res[l]                                                                # (b*winnum) * n-1 * 2
                        wanomaly_map = calWinAnoMap(winmaps, win_masklist, args.winsize, 
                                                    int(args.image_size / args.patchsize), device, 
                                                    args.usewinimgencoder, convResModule, args.winZipSize)      # b * 37 * 37
                        # print('anomaly map size: {}'.format(wanomaly_map.shape))
                        win_img_pr += wanomaly_map.max()
                        
                        wanomaly_map = F.interpolate(wanomaly_map.unsqueeze(1), size=calimg_size, mode='bilinear', align_corners=True)             # b * 1 * 240 * 240
                        win_anomaly_maps.append(wanomaly_map.squeeze(1).cpu().numpy())                                                # l * b * imgsize * imgsize
                    
                    win_anomaly_map = np.sum(win_anomaly_maps, axis=0) / s
                    win_img_pr = win_img_pr / s
                    
                    if args.winmode == 'mul':
                        ww = args.winwight
                        ow = 1 - ww
                        # avg
                        img_pr_ = ow * img_pr + ww * win_img_pr
                        anomaly_map = ow * anomaly_map + ww * win_anomaly_map
                        # img_pr_ = torch.tensor(anomaly_map.max())
                        
                        # max
                        # anomaly_map = np.concatenate((anomaly_map, win_anomaly_map), axis=0)     # 2 * 240 * 240
                        # anomaly_map = np.max(anomaly_map, axis=0)
                        # anomaly_map = np.expand_dims(anomaly_map, axis=0)
                        # img_pr_ = anomaly_map.max()
                        # img_pr_ = torch.tensor(img_pr_)
                        
                        results['pr_sp'].append(img_pr_.cpu().item())
                        sprint('anomaly map size: {}'.format(anomaly_map.shape))
                        results['anomaly_maps'].append(anomaly_map)  
                        
                    if args.winmode == 'onlywin':
                        anomaly_map = win_anomaly_map                        
                        results['pr_sp'].append(win_img_pr.cpu().item())
                        results['anomaly_maps'].append(anomaly_map)                
            
            end_time = time.perf_counter()
            use_time = end_time - start_time
            print('[time] use time : {} s'.format(use_time))
            use_time_micro = use_time * 1e6
            avg_use_time_micro += use_time_micro
            
            sprint('anomap max: {}, anomap min: {}'.format(anomaly_map.max(), anomaly_map.min()))
            # count res
            if args.count_res:
                sprint('anomap max: {}, anomap min: {}'.format(anomaly_map.max(), anomaly_map.min()))
                rrcount(torch.tensor(anomaly_map), gt_mask)
                sprint('cos_res shape: {}'.format(cos_res.shape))
                sprint('cos_res max: {}, cos_res min: {}'.format(cos_res.max(), cos_res.min()))
                coscount(cos_res, gt_mask)

            results['img_path'].append(items['img_path'][0])
            # visualization
            if args.saveimg:
                path = items['img_path']                
                cls = path[0].split('/')[-2]
                filename = path[0].split('/')[-1]
                vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (calimg_size, calimg_size)), cv2.COLOR_BGR2RGB)  # RGB
                # mask = normalize(anomaly_map[0]) 
                mask = anomaly_map[0]
                vis_ = np.copy(vis)           
                vis = apply_ad_scoremap(vis, mask)            
                gt_vis = apply_ad_scoremap(vis_, gt_mask[0,0,:,:].numpy())
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
                gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR)
                save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)            
                if not os.path.exists(save_vis):
                    os.makedirs(save_vis)
                
                cv2.imwrite(os.path.join(save_vis, filename), vis)
                cv2.imwrite(os.path.join(save_vis, filename.split('.')[0]+'_gt.'+filename.split('.')[1]), gt_vis)


    avg_use_time_micro = avg_use_time_micro / len(test_data)
    logger.info("\nAverage time per image %s microsecond", str(avg_use_time_micro))
    
    # save count res
    if args.count_res:
        rrcount.saveCountRes()
        coscount.saveCountRes()
    
    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    for obj in obj_list:
        print('{} start cal'.format(obj))
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        img_srcs = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
                img_srcs.append(results['img_path'][idxes])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)
        if args.mode == 'few_shot':
            pr_sp_tmp = np.array(pr_sp_tmp)
            pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
            pr_sp = 0.5 * (pr_sp + pr_sp_tmp)

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        th = np.insert(thresholds, 0, 0)
        th = th[np.isfinite(f1_scores)]
        f1_scores = f1_scores[np.isfinite(f1_scores)]
        f1_max_th = th[np.argmax(f1_scores)]
        print('f1_max_th is : {}'.format(f1_max_th))
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = 0.0
        # aupro = cal_pro_score(gt_px, pr_px)

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)
        obj_res = "{}: \nauroc_sp: {}, auroc_px: {}, f1_sp: {}, f1_px: {}, aupro: {}, ap_sp: {}, ap_px: {}".format(
            obj, auroc_sp, auroc_px, f1_sp, f1_px, aupro, ap_sp, ap_px
        )
        
        # save f1-max pic
        for img_src, maskmap in zip(img_srcs, pr_px):
            cls = img_src.split('/')[-2]
            filename = img_src.split('/')[-1]
            
            pic_src = cv2.cvtColor(cv2.resize(cv2.imread(img_src), (calimg_size, calimg_size)), cv2.COLOR_BGR2RGB)  # RGB
            maskmap[maskmap > f1_max_th] = 1.
            maskmap[maskmap <= f1_max_th] = 0.
            f1_vis = apply_ad_scoremap(pic_src, maskmap)
            f1_vis = cv2.cvtColor(f1_vis, cv2.COLOR_RGB2BGR)
            
            save_vis = os.path.join(save_path, 'imgs', obj, cls)
            cv2.imwrite(os.path.join(save_vis, filename.split('.')[0]+'_F1-max.'+filename.split('.')[1]), f1_vis)
            
        
        # print(obj_res)
        print("{} metrics cal over".format(obj))
        # logger.info(obj_res)
        

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/data/wangzhehong/database/VISA", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--rrcount_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--coscount_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./exps/vit_huge_14/model_epoch12.pth', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--calImage_size", type=int, default=240, help="cal image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")    
    parser.add_argument("--saveimg", type=int, default=1, help="saveimg?")
    parser.add_argument("--count_res", type=int, default=0, help="count res?")
    parser.add_argument("--useNorm", type=int, default=0, help="use norm?")
    
    parser.add_argument("--useAdapter", type=int, default=0, help="use adapter?")
    parser.add_argument("--useAbs", type=int, default=0, help="use diff abs?")
    parser.add_argument("--useFirstBatchNorm", type=int, default=0, help="use firstbatchnorm?")
    parser.add_argument("--useConvRes", type=int, default=0, help="use NAM?")
    parser.add_argument("--useSENet", type=int, default=0, help="use SENet?")
    parser.add_argument("--useZeroCls", type=int, default=0, help="use zero cls?")
    parser.add_argument("--ResFusionMode", type=str, default='conv', help="resfusion: conv, avg, nan")
    parser.add_argument("--RRmode", type=str, default='fewshot', help="mode: fewshot, zeroshot")
    parser.add_argument("--alignMode", type=str, default='mulmax', help="alignmode")
    parser.add_argument("--useImgIdx", type=int, default=0, help="use imgidx mem bank")
    
    parser.add_argument("--winmode", type=str, default='nan', help="nan, mul, onlywin")
    parser.add_argument("--winsize", type=int, default=5, help="winsize,unit:patch,14px")
    parser.add_argument("--winZipSize", type=int, default=5, help="winzipsize,unit:,14px")
    parser.add_argument("--winstride", type=int, default=1, help="winstride,unit:,14px")
    parser.add_argument("--patchsize", type=int, default=14, help="patchsize,14px")
    parser.add_argument("--winuseConvRes", type=int, default=0, help="win use NAM?")
    parser.add_argument("--usewinmem", type=int, default=1, help="use win mem?")
    parser.add_argument("--usewinimgencoder", type=int, default=1, help="use win F(*)?")
    parser.add_argument("--winwight", type=float, default=0.5, help="win weight")
    # few shot
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    test(args)
