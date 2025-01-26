import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise

import logging

import open_clip
from dataset import VisaDataset, MVTecDataset
from model import LinearLayer
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble import encode_text_with_prompt_ensemble

from few_shot import memory, memory_win

from FASLModel import FASL, winSplit



def visa_cropImg(img, imggt, imgsize = 518, cropplus_rate = 0.5):           # 3 * imgsize * imgszie, 1 * imgsize * imgsize
    # visa treansform
    visa_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((100, 100)),
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor()
    ])
    
    visa_transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((50, 50)),
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor()
    ])
    
    imgwithgt = torch.cat((imggt, img), dim=0)
    
    if imggt.max() < 0.5:
        random_number = random.random()
        if random_number < 0.5:
            imgtt = visa_transform(imgwithgt)
            img_, gt_ = imgtt[1:, :, :], imgtt[0, :, :]
            gt_ = gt_.unsqueeze(0)
        else:
            img_, gt_ = img, imggt
        
        return img_, gt_
        
    random_number = random.random()
    if random_number < cropplus_rate:
        while True:
            cropimg = visa_transform_(imgwithgt)
            img_ = cropimg[1:, :, :]
            gt_ = cropimg[0, :, :]
            if gt_.max() > 0.5:
                break
    else:
        while True:
            cropimg = visa_transform(imgwithgt)
            img_ = cropimg[1:, :, :]
            gt_ = cropimg[0, :, :]
            if gt_.max() > 0.5:
                break
            
    gt_ = gt_.unsqueeze(0) 
    return img_, gt_
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # configs
    dataset_dir = args.train_data_path
    dataset_name = args.dataset
    few_shot_features = args.few_shot_features
    
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model.eval().to(device)
    # tokenizer = open_clip.get_tokenizer(args.model)
    
    # winSpliter
    winSpliter = winSplit(winsize=args.winsize, stride=args.winstride)
    
    if args.winmode != 'nan':
        model_win, _, _ = open_clip.create_model_and_transforms(args.model, args.winsize*args.patchsize, pretrained=args.pretrained)
        model_win.eval().to(device)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
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
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                  aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    obj_list = train_data.get_cls_names()

    # few shot    
    mem_features = memory(args.model, model, obj_list, dataset_dir, save_path, preprocess, transform,
                              args.k_shot, few_shot_features, dataset_name, device, args.useNorm)
    # win few shot
    if args.winmode != 'nan':
        win_mem_features = memory_win(args.model, model_win, obj_list, dataset_dir, save_path, preprocess, transform,
                              args.k_shot, few_shot_features, dataset_name, device, win_size=args.winsize, win_stride=args.winstride)
    
    # linear layer
    # trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
    #                               len(args.features_list), args.model).to(device)
    
    # selfModule
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
    RR = RR.train().to(device)

    optimizer = torch.optim.Adam(list(RR.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        obj_list = train_data.get_cls_names()
        # text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
        
    
    for epoch in range(epochs):
        print("{} / {} ===================================\n ".format(epoch, epochs))
        loss_list = []
        iidx = 0
        
        for items in train_dataloader:
            # if iidx == 1:
            #     break          
            iidx += 1
            image = items['img'].to(device)  
            gts = items['img_mask'].to(device)                                  # b * 1 * imgsize * imgsize
            print("gts shape: {}".format(gts.shape))
            cls_name = items['cls_name']
            img_path = items['img_path']
            
            print("image size: {}".format(image.shape))                         # b * 3 * imgsize * imgsize
            print("cls_name size: {}".format(len(cls_name)))                  
            
            if args.winmode != 'nan':
                win_image = image.clone()
                win_gt = gts.clone()
                
                win_images, win_masklist = winSpliter(win_image)                # (winnum * b) * 3 * winszie * winszie
                win_gts, _ = winSpliter(win_gt)                                 # (winnum * b) * 1 * winszie * winszie
            
            # visa crop
            if args.dataset == 'visa' and args.use_crop:
                imgs = []
                imggts = []
                for img, g in zip(image, gts):                    
                    # print("g shape: {}".format(g.shape))                      # 1 * imgsize * imgszie
                    random_number = random.random()
                    if random_number < args.crop_rate:
                        img, g = visa_cropImg(img, g, cropplus_rate=args.crop_rate_plus)
                        img = img.to(device)
                        g = g.to(device)
                    imgs.append(img)
                    imggts.append(g)
                image = torch.stack(imgs)
                gts = torch.stack(imggts)
                
            
            with torch.cuda.amp.autocast():

                # few shot   
                with torch.no_grad():
                    image_features, _patch_tokens = model.encode_image(image, few_shot_features)
                    if args.useNorm:
                        _patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in _patch_tokens]
                    
                    if args.winmode != 'nan':
                        win_image_features, win_patch_tokens = model_win.encode_image(win_images, few_shot_features)
                                  
                # _patch_tokens = [pp.requires_grad = True for pp in _patch_tokens]             
                anomaly_maps_few_shot = []
                print("patch tokens size: {} * {}".format(len(_patch_tokens), _patch_tokens[0].shape))      # layernum * b * n * c
                
                few_features = []
                for idx, p in enumerate(_patch_tokens):                                         # layer     # p: b*n*c
                    print(" ============ layer_{} ============== ".format(idx))
                    few_feature = []
                    for b in range(len(cls_name)):                                                 # batchsize
                        print(" ---- {} ---- ".format(b))
                        print("cls name: {}".format(cls_name[b]))
                        print("img path: {}".format(img_path[b]))
                        mem_feature = mem_features[cls_name[b]]
                        mem_feature_l = mem_feature[idx]
                        if 'ViT' in args.model:
                            p_ = p[b, 1:, :]                                                     # n * c
                        else:
                            p_ = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()    
                            
                        if args.alignMode == 'diff2':
                            # n, c = p_.shape
                            # nm, _ = mem_feature_l.shape
                            # p_ = p_.unsqueeze(1).expand(n, nm, c)
                            # p_ = p_ - mem_feature_l                     # n * nm * c
                            # p_ = p_.norm(dim=2)                         # n * nm
                            # m_idx = torch.argmax(p_, dim=1)             # n
                            # few_features_l = mem_feature_l[m_idx]       # n * c
                            
                            few_features_l = []
                            for ft in p_:           # ft: c
                                ft = mem_feature_l - ft             # n_ * c
                                ft = ft.norm(dim=1)                 # n_
                                m_idx = torch.argmin(ft)
                                few_features_l.append(mem_feature_l[m_idx]) 
                            few_features_l = torch.stack(few_features_l)        # n * c
                            print("few_features_l size: {}".format(few_features_l.shape))
                        
                        if args.alignMode == 'mulmax':                        
                            print("mem size : {}".format(mem_feature_l.shape))
                            print("p size: {}".format(p_.shape))
                            m = p_ @ mem_feature_l.T                            
                            print("m size: {}".format(m.shape))
                            m_idx = torch.argmax(m, dim=1)
                            print("m_idx size: {}".format(m_idx.shape))                            
                            few_features_l = []
                            for ii in m_idx:
                                few_features_l.append(mem_feature_l[ii])
                                
                            few_features_l = torch.stack(few_features_l)                            # n * c
                            print("few_features_l size: {}".format(few_features_l.shape))
                            
                        few_feature.append(few_features_l)
                        
                        print("\n")
                    
                    few_feature = torch.stack(few_feature)                                      # b * n * c
                    few_features.append(few_feature)
                few_features = torch.stack(few_features)                                        # l * b * n * c
                print("few_features size: {} * {}".format(len(few_features), few_features[0].shape))
                few_features = few_features.to(device)
                
                few_features.requires_grad = True
                _patch_tokens = torch.stack(_patch_tokens)
                _patch_tokens.requires_grad = True
                
                #
                res, fusion_res = RR(_patch_tokens, few_features)                                               # l * b * n * 2
                
                anomaly_maps = []
                for l in range(len(res)):
                    anomaly_map = res[l]                                                            # b * n * 2
                    B, N, C = anomaly_map.shape
                    H = int(np.sqrt(N))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=image_size, mode='bilinear', align_corners=True)
                    anomaly_maps.append(anomaly_map)                                                # l * b * 2 * imgsize * imgsize
                
                print("anomaly maps size: {} * {}".format(len(anomaly_maps), anomaly_maps[0].shape))
                
                fusion_anomaly_maps = []
                for l in range(len(fusion_res)):
                    fanomaly_map = fusion_res[l]
                    B, N, C = fanomaly_map.shape
                    H = int(np.sqrt(N))
                    fanomaly_map = F.interpolate(fanomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=image_size, mode='bilinear', align_corners=True)
                    fusion_anomaly_maps.append(fanomaly_map)
                            


            # losses
            gt = gts.squeeze()                                                  # b * 3 * imgsize * imgsize      
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            gt.requires_grad = True
            loss = 0
            for num in range(len(anomaly_maps)):                                # layer
                loss += loss_focal(anomaly_maps[num], gt)
                loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
                
            # loss+
            for num in range(len(fusion_anomaly_maps)):
                loss += loss_focal(fusion_anomaly_maps[num], gt)
                loss += loss_dice(fusion_anomaly_maps[num][:, 1, :, :], gt)

            print("\n ********************** batch loss: {} **************************\n".format(loss.item()))
            # before_param = next(RR.parameters()).data.clone()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            


        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_RR': RR.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/vit_large_14_518', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9],
                        help="features used for few shot")
    
    parser.add_argument("--useAdapter", type=int, default=0, help="use adapter?")
    parser.add_argument("--useAbs", type=int, default=0, help="use diff abs?")
    parser.add_argument("--useFirstBatchNorm", type=int, default=0, help="use firstbatchnorm?")
    parser.add_argument("--useConvRes", type=int, default=0, help="use NAM?")
    parser.add_argument("--useSENet", type=int, default=0, help="use senet?")
    parser.add_argument("--useZeroCls", type=int, default=0, help="use zero cls?")
    parser.add_argument("--useForeHence", type=int, default=0, help="***")
    parser.add_argument("--ResFusionMode", type=str, default='conv', help="res fusion: conv, avg, nan")
    parser.add_argument("--RRmode", type=str, default='fewshot', help="mode: fewshot, zeroshot")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="mvtec aug rate")
    parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--use_crop", type=int, default=0, help="visa crop?")
    parser.add_argument("--crop_rate", type=float, default=0.6, help="visa crop rate")
    parser.add_argument("--crop_rate_plus", type=float, default=0.6, help="visa crop rate")
    parser.add_argument("--alignMode", type=str, default='mulmax', help="align mode")
    parser.add_argument("--useNorm", type=int, default=0, help="use norm?")
    
    parser.add_argument("--winmode", type=str, default='nan', help="nan, mul, onlywin")
    parser.add_argument("--winsize", type=int, default=5, help="winsize,unit:patch,14px")
    parser.add_argument("--winstride", type=int, default=2, help="winstride,unit:patch,14px")
    parser.add_argument("--patchsize", type=int, default=14, help="patchsize,14px")
    
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    setup_seed(111)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    train(args)

