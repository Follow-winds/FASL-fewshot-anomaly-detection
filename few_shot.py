import torch
from dataset import VisaDataset, MVTecDataset
from selfModel import winSplit
from torch.nn import functional as F



def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot,
           few_shot_features, dataset_name, device, useNorm=False):
    mem_features = {}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                if useNorm:
                    patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in patch_tokens]
                if 'ViT' in model_name:
                    patch_tokens = [p[0, 1:, :] for p in patch_tokens]                      # l * [n-1 * c]
                else:
                    patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features



def memory_win(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot,
           few_shot_features, dataset_name, device, winzsize, win_size=5, win_stride=1, usewinimgencoder=1, image_size=518, useNorm=False):
    winspliter = winSplit(winsize=win_size, stride=win_stride, winZipSize=winzsize)
    mem_features = {}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in dataloader:
            _image = items['img'].to(device)                                                # b * c * h * h
            with torch.no_grad():
                images, _ = winspliter(_image)                                               # (winnum * b) * c * h_ * h_
            if usewinimgencoder == 0:
                images = F.interpolate(images, size=image_size, mode='bilinear', align_corners=True)
            with torch.no_grad():
                for image in images:
                    image = image.unsqueeze(0)
                    image_features, patch_tokens = model.encode_image(image, few_shot_features) # l * (winnum*b) * n * c
                    if useNorm:
                        patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in patch_tokens]
                    image_features = image_features.cpu()                    
                    if 'ViT' in model_name:
                        patch_tokens = [p[:, 1:, :] for p in patch_tokens]                      # l * (winnum*b) * n-1 * c
                    else:
                        patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                    
                    patch_tokens = torch.stack(patch_tokens)
                    patch_tokens = patch_tokens.permute(1,0,2,3)                                # (winnum*b) * l * n-1 * c
                    patch_tokens = patch_tokens.cpu()
                    for pts in patch_tokens:
                        t = [pt for pt in pts]
                        features.append(t)
                        
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features

# img feature index memory
def memory_imgidx(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot,
           few_shot_features, dataset_name, device, useNorm=False):
    mem_features = {}
    mem_img_features = {}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        img_features = []
        for items in dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                image_features = F.normalize(image_features, p=2, dim=-1)                   # 1 * 768
                image_features = image_features.squeeze(0)                                  # 768
                
                if useNorm:
                    patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in patch_tokens]
                if 'ViT' in model_name:
                    patch_tokens = [p[0, 1:, :] for p in patch_tokens]                      # l * [n-1 * c]
                else:
                    patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens)                                               # k * l * [n-1 * c]
                img_features.append(image_features)
        mem_img_features[obj] = torch.stack(img_features)                                   # k * c
        mem_features[obj] = features                                                        # k * l * n-1 * c
        # mem_features[obj] = [torch.cat([features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    print('img mem features size: {}'.format(torch.stack(img_features).shape))
    return mem_img_features, mem_features

# win img feature index memory
def memory_win_imgidx(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot,
           few_shot_features, dataset_name, device, winzsize, win_size=5, win_stride=1, usewinimgencoder=1, image_size=518, useNorm=False):
    winspliter = winSplit(winsize=win_size, stride=win_stride, winZipSize=winzsize)
    mem_features = {}
    mem_img_features = {}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = [] 
        img_features = []       
        for items in dataloader:
            _image = items['img'].to(device)                                                # b * c * h * h
            with torch.no_grad():
                images, msklist = winspliter(_image)                                               # (winnum * b) * c * h_ * h_
                winnum = len(msklist)
            if usewinimgencoder == 0:
                images = F.interpolate(images, size=image_size, mode='bilinear', align_corners=True)
            with torch.no_grad():
                for image in images:                                                    
                    image = image.unsqueeze(0)
                    image_features, patch_tokens = model.encode_image(image, few_shot_features) # l * 1 * n * c
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    image_features = image_features.squeeze(0)
                    img_features.append(image_features)                                         # (winnum * k) * c
                    
                    if useNorm:
                        patch_tokens = [F.normalize(_p, p=2, dim=-1) for _p in patch_tokens]
                    image_features = image_features#.cpu()                    
                    if 'ViT' in model_name:
                        patch_tokens = [p[:, 1:, :] for p in patch_tokens]                      # l * 1 * n-1 * c
                    else:
                        patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                    
                    patch_tokens = torch.stack(patch_tokens)
                    patch_tokens = patch_tokens.permute(1,0,2,3)                                # 1 * l * n-1 * c
                    patch_tokens = patch_tokens.cpu()
                    # print('win patch_tokens size: {}'.format(patch_tokens.shape))
                    for pts in patch_tokens:                                                    # l * n- 1 * c
                        features.append(pts)                                                    # (winnum * k) * l * n-1 * c
        features = torch.stack(features)
                # _, L, N, C = features.shape
                # features = features.view(winnum, -1, L, N, C)
                # features = features.permute(1, 2, 0, 3, 4)                                      # k * l * winnum * n-1 * c
                # features = features.view(-1, L, winnum*N, C)                                    # k * l * (winnum * n-1) * c
        mem_features[obj] = features                                                        # (winnum * k) * l * n-1 * c
        mem_img_features[obj] = torch.stack(img_features)                                   # (winnum * k) * c
        # mem_features[obj] = [torch.cat(
        #     [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    print('win img mem features size: {}'.format(torch.stack(img_features).shape))
    print('win mem features size: {}'.format(features.shape))
    return mem_img_features, mem_features