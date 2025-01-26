import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class forGroundEnHance(nn.Module):
    def __init__(self, mapsize=37, kernel_size=5):
        super(forGroundEnHance, self).__init__()
        
        self.mapsize = mapsize
        self.kernel_size = kernel_size
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(1024, 512, kernel_size, padding='same', stride=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size, padding='same', stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size, padding='same', stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size, padding='same', stride=1)       
        
    def getMap(self, map):     # b * n * c
        B, N, C = map.shape
        H = int(np.sqrt(N))
        map = map.permute(0, 2, 1)  # b * c * n
        map = map.view(B, C, H, H)  # b * c * h * h
        
        y = self.conv(map)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.conv4(y)
        y = self.sigmoid(y)     # b * 1 * h * h
        y = y.view(B, 1, -1)
        y = y.permute(0, 2, 1) # b * n * 1
        y = y.squeeze(-1)       # b * n
        
        return y
    
    def forward(self, map, hancemap):
        '''
            map: b * n * 2
            hancemap: b * n
        '''
        
        anoMap = map[:, :, 1]               # b * n        
        anoMap = anoMap * hancemap          # b * n
        resmap = [1. - anoMap, anoMap]
        resmap = torch.stack(resmap)        # 2 * b * n        
        resmap = resmap.permute(1, 2, 0)
        
        return resmap                       # b * n * 2
        


class countRes(nn.Module):
    
    def __init__(self, savepath):
        super(countRes, self).__init__()
        
        # 0-0.1 0.1-0.2 0.2-0.3 0.3-0.4 0.4-0.5 0.5-0.6 0.6-0.7 0.7-0.8 0.8-0.9 0.9-1
        self.anoCount = torch.tensor([0,0,0,0,0,0,0,0,0,0])
        self.norCount = torch.tensor([0,0,0,0,0,0,0,0,0,0])
        
        self.savepath = savepath
        
    def forward(self, resmap, gtmap):
        '''
            resmap: b * calsize * calsize
            gtmap: b * 1 * calsize * calsize
        '''
        b, _, _ = resmap.shape
        gtmap = gtmap.squeeze(1)                            # b * calsize * calsize
        resmap = resmap.view(b, -1)                         # b * (cal*cal)
        gtmap = gtmap.view(b, -1)                           # b * (cal*cal)
        b, n = gtmap.shape
        for i in range(b):
            for j in range(n):
                if gtmap[i,j] > 0.5:
                    self.anoCount[int(resmap[i,j] / 0.1)] += 1
                else:
                    self.norCount[int(resmap[i,j] / 0.1)] += 1
                    
    def saveCountRes(self):
        with open(self.savepath, 'w') as file:            
            file.write('anoCount: ')
            for num in self.anoCount:
                file.write(str(num.item()))
                file.write(', ')
            file.write('\n')
            file.write('norCount: ')
            for num in self.norCount:
                file.write(str(num.item()))
                file.write(', ')
            file.write('\n')


    
class SELayer(nn.Module):
    def __init__(self, channel=1024, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, c = x.shape        
        y = x.mean(dim=1)               # b * c        
        y = self.fc(y)                    # b * c   
        y = y.unsqueeze(1)     
        return x * y.expand_as(x)

class clsModule(nn.Module):
    def __init__(self, firstbatchnorm = True, channel = 1024):
        super(clsModule, self).__init__()
        
        self.firstbatchnorm = firstbatchnorm
        
        self.fbatchNorm = nn.BatchNorm1d(1024)
        self.proj = nn.Sequential(            
            nn.Linear(channel, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax()
        )
        
    def forward(self, x):        
        if self.firstbatchnorm:
            x = self.fbatchNorm(x)
            
        y = self.proj(x)
        return y

class adapter(nn.Module):
    def __init__(self, in_channel = 1024, out_chanel=1024):
        super(adapter, self).__init__()
        self.in_c = in_channel
        self.out_c = out_chanel
        
        self.l = nn.Linear(self.in_c, 256, bias=True)
        self.l2 = nn.Linear(256, self.out_c, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.l(x)
        y = self.relu(y)
        y = self.l2(y)
        return y

# NAM
class resConv(nn.Module):
    def __init__(self, imgsize = 37, in_kernel_size = 5, out_kernel_size = 5):
        super(resConv, self).__init__()
        self.imgsize = imgsize
        self.in_kernel_size = in_kernel_size
        self.out_kernel_size = out_kernel_size
        
        self.conv = nn.Conv2d(2, 6, in_kernel_size, padding='same', stride=1)
        self.conv2 = nn.Conv2d(6, 2, out_kernel_size, padding='same', stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    
    def forward(self, x):                                   # b * n * 2           
        B, N, C = x.shape
        x = x.permute(0, 2 ,1)                              # b * 2 * n
        H = int(np.sqrt(x.shape[2]))
        x = x.view(-1, 2, H, H)                             # b * 2 * 37 * 37        
        
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.softmax(x)
        
        x = x.view(B, 2, N)
        x = x.permute(0, 2, 1)                              # b * n * 2
        return x

class resFusion(nn.Module):
    def __init__(self, imgsize = 37, kernel_size = 5):
        super(resFusion, self).__init__()
        self.kernel_size = kernel_size
        self.imgsize = imgsize
        
        self.conv = nn.Conv2d(4, 12, kernel_size, padding='same', stride=1)
        self.conv2 = nn.Conv2d(12, 2, kernel_size, padding='same', stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, zero, few):                                   # b * n * 2 
        B, N, C = zero.shape
        few = few.permute(0, 2, 1)                                  # b * 2 * n
        zero = zero.permute(0, 2, 1)
        few = few.view(-1, 2, self.imgsize, self.imgsize)           # b * 2 * 37 * 37
        zero = zero.view(-1, 2, self.imgsize, self.imgsize)
        
        inmap = torch.cat((zero, few), dim=1)                       # b * 4 * 37 * 37
        y = self.conv(inmap)
        y = self.relu(y)
        y = self.conv2(y)                                      
        outmap = self.softmax(y)                                    # b * 2 * 37 * 37
        outmap = outmap.view(B, 2, N)
        outmap = outmap.permute(0, 2, 1)                            # b * n * 2
        return outmap

class zeroCls(nn.Module):
    def __init__(self, useFirstBatchNorm = False, layerNum = 4):
        super(zeroCls, self).__init__()
        
        self.layerNum = layerNum
        
        # Zero classifier
        self.zero_clsList = nn.ModuleList([clsModule(firstbatchnorm=useFirstBatchNorm) for i in range(layerNum)])
        
        # nam
        # self.convRes = resConv()
        
    def forward(self, features):
        '''
            features:         l * b * n * c
        '''
        if self.layerNum != len(features):
            raise ValueError("The middle layers are not aligned. Please check the layers set by the model")
        
        features = [f[:, 1:, :] for f in features]
        anomalyMap = []
        for l in range(self.layerNum):
            fl = features[l]                                                    # b * n * c
            B, N, C = fl.shape
            fl = fl.contiguous().view(-1, C)                                    # (b*n) * c
            anomalyMap_l = self.zero_clsList[l](fl)                             # (b*n) * 2
            anomalyMap_l = anomalyMap_l.view(B, N, -1)                          # b * n * 2
            anomalyMap.append(anomalyMap_l)
            
            # # nam
            # if self.useConvRes:                
            #     anomalyMap_l_ = anomalyMap_l.clone()
            #     anomalyMap_l_ = self.convRes(anomalyMap_l_)
            #     anomalyMap.append(anomalyMap_l_)
            
        return anomalyMap                                                       # l * b * n * 2 
            


# FASL: FDM+ASLM
class FASL(nn.Module):
    def __init__(self, 
                 useZeroCls = False, 
                 useAdapter = False, 
                 useAbs = True, 
                 useFirstBatchNorm = True, 
                 useConvRes = False, 
                 useSENet = False, 
                 usefgHance = False, 
                 ResFusionMode = 'conv', 
                 layerNum = 4, 
                 mode = 'fewshot'):

        super(FASL, self).__init__()
        self.layerNum = layerNum
        self.useAbs = useAbs
        self.useConvRes = useConvRes
        self.useAdapter = useAdapter
        self.useZeroCls = useZeroCls
        self.ResFusionMode = ResFusionMode
        self.useSENet = useSENet
        self.usefgHance = usefgHance
        
        self.mode = mode

        # senet
        self.rAttn = SELayer()
        
        # zero cls
        self.zeroClsModule = zeroCls()
        
        # adapter
        self.adapterList = nn.ModuleList([adapter() for i in range(layerNum)])
        
        # acls
        self.clsList = nn.ModuleList([clsModule(firstbatchnorm=useFirstBatchNorm) for i in range(layerNum)])
                
        # nam
        self.convRes = resConv()
        
        # res fusion
        self.resfus = resFusion()
        
    # return nam
    def getConvRes(self):
        return self.convRes
        
        
    def forward(self, features, few_features):
        '''
            features:             l * b * n * c
            few_features:         l * b * n * c
        '''
        if self.layerNum != len(features):
            raise ValueError("please check layernum!")
        
        if self.mode == 'zeroshot':
            self.useZeroCls = True
        
        anomalyMap = []
        
        # zeroshot *************************************
        if self.useZeroCls:
            zero_anomalyMap = self.zeroClsModule(features)                              # l * b * n * 2
                        
            for zero_anomalyMap_l in zero_anomalyMap:
                anomalyMap.append(zero_anomalyMap_l)
                if self.useConvRes:
                    zero_anomalyMap_l_ = zero_anomalyMap_l.clone()
                    zero_anomalyMap_l_ = self.convRes(zero_anomalyMap_l_)
                    anomalyMap.append(zero_anomalyMap_l_)
                    
        if self.mode == 'zeroshot':
            return anomalyMap, []       
        
        
        # fewshot *************************************
        features = [f[:, 1:, :] for f in features]
        
        for l in range(self.layerNum):
            # use adapter ---------------------------------------------------------------------------
            if self.useAdapter:
                fl = self.adapterList[l](features[l])
                ffl = self.adapterList[l](few_features[l])                
            else:
                fl = features[l]                        # b * n * c
                ffl = few_features[l]                
                

            # abs ---------------------------------------------------------------------------
            if self.useAbs:
                residual_l = torch.abs(fl - ffl)
            else:
                residual_l = fl - ffl                      # b * n * c
                
            # senet
            if self.useSENet:
                residual_l = self.rAttn(residual_l)         # b * n * c
            
            # cls -------------------------------------------------------------------------------
            B, N, C = residual_l.shape
            # print(residual_l.shape)
            residual_l = residual_l.contiguous().view(-1, C)
            anomalyMap_l = self.clsList[l](residual_l)
            # print("anomalyMap_l size: {}".format(anomalyMap_l.shape))
            anomalyMap_l = anomalyMap_l.view(B, N, -1)                          # b * n * 2
            # print("anomalyMap_l view size: {}".format(anomalyMap_l.shape))
            # print("softmax res: {}".format(anomalyMap_l[0][0]))

            anomalyMap.append(anomalyMap_l)
            
            # nam ---------------------------------------------------------------------------
            if self.useConvRes:
                # print("pre size: {}".format(anomalyMap_l.shape))
                anomalyMap_l_ = anomalyMap_l.clone()
                anomalyMap_l_ = self.convRes(anomalyMap_l_)

                anomalyMap.append(anomalyMap_l_)
        
        # res fusion
        fusion_anomalyMap = []
        if self.mode == 'fewshot' and self.useZeroCls:
            dis = self.layerNum
            start = 0
            stride = 1
            if self.useConvRes:
                dis = dis * 2
                start = 1
                stride = 2
                        
            if self.ResFusionMode == 'conv':
                for ll in range(self.layerNum):
                    zero_m_l = anomalyMap[start + ll * stride]                      # b * n * 2
                    few_m_l = anomalyMap[start + ll * stride + dis]                 # b * n * 2
                    
                    fusion_map_l = self.resfus(zero_m_l, few_m_l)                   # b * n * 2
                    fusion_anomalyMap.append(fusion_map_l)                          # l * b * n * 2
                    
            elif self.ResFusionMode == 'avg':
                for ll in range(self.layerNum):
                    zero_m_l = anomalyMap[start + ll * stride]                      # b * n * 2
                    few_m_l = anomalyMap[start + ll * stride + dis]                 # b * n * 2
                    
                    fusion_map_l = (zero_m_l + few_m_l) / 2                         # b * n * 2
                    fusion_anomalyMap.append(fusion_map_l)                          # l * b * n * 2                
            else:                
                print('no res fusion')
                
        return anomalyMap, fusion_anomalyMap                                             # l * b * n * 2
    
    
    
    
class winSplit(nn.Module):
    def __init__(self, imgsize = 518, patchsize = 14, winsize = 5, stride = 1, winZipSize = 7):
        '''
            imgsize: The size of the image to be split
            patchsize: The size of the segmentation unit
            winsize: Window size (unit: patch)
            stride: Slide step, unit patch
            winZipSize: Window compression size, in patch
        '''
        super(winSplit, self).__init__()
        self.imgsize = imgsize
        self.patchsize = patchsize
        self.winsize = winsize
        self.stride = stride
        self.winZipSize = winZipSize
        
        self.masklist, self.pmasklist = self.makemask()
        
    def makemask(self):
        # patch_H = int(self.imgsize / self.patchsize)
        winsize = self.winsize * self.patchsize
        stride = self.stride * self.patchsize
        
        masklist = []
        pmasklist = []
        for row in range(0, self.imgsize - winsize + 1, stride):
            for col in range(0, self.imgsize - winsize + 1, stride):
                masklist.append([row, row + winsize, col, col + winsize])
                pmasklist.append([int(row / self.patchsize), int((row + winsize) / self.patchsize),
                                  int(col / self.patchsize), int((col + winsize) / self.patchsize)])
                
        return masklist, pmasklist
        
    def forward(self, x):                           
        # B, C, H, _ = x.shape        
        x_ = []
        for mask in self.masklist:
            row_start = mask[0]
            row_end = mask[1]
            col_start = mask[2]
            col_end = mask[3]
            
            x__ = x[:, :, row_start:row_end, col_start:col_end]                  # b * c * winsize * winsize
            x_.append(x__)
            
        x_ = torch.stack(x_)                                                    # winnum * b * c * winszie(13*14) * winszie
        _, B, C, H, _ = x_.shape
        x_ = x_.view(-1, C, H, H)                                               # (winnum * b) * c * winszie * winszie
        
        if self.winZipSize != self.winsize:
            winzipsize = self.winZipSize * self.patchsize
            x_ = F.interpolate(x_, size=winzipsize, mode='bilinear', align_corners=True)        # (winnum * b) * c * winzipsize * winzipsize
        
        return x_, self.pmasklist