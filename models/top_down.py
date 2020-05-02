import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils


class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        # print (x.shape,"x.shape")
        out = self.conv1(x)
        # print (out.shape,"out.shape")
        out = self.conv2(out)
        # print (out.shape,"out.shape")
        out = self.conv3(out)
        # print (out.shape,"out.shape")
        out = self.conv4(out)
        # print (out.shape,"out.shape")
        out = self.conv5(out)
        # print (out.shape,"out.shape")
        out = self.conv6(out)
        # print (out.shape,"out.shape")
        out_feat = self.conv7(out)
        # print (out.shape,"out.shape")
        # print ("+"*100)


        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.view(-1)
        return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        ## addding one more conv layer to make the normal image of shpae [b,3,32,32]
        ### formula referred from rockson's answer @ https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer 
        self.deconv4= model_utils.conv(batchNorm, 3, 3,  k=(65,1), stride=1, pad=0)

        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        # print (x.shape,"x.shape")
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        # print (x.shape,"x.shape")
        out    = self.deconv1(x)
        # print (out.shape,"out.shape")
        out    = self.deconv2(out)
        # print (out.shape,"out.shape")
        out    = self.deconv3(out)
        # print (out.shape,"out.shape")
        
        normal = self.est_normal(out)
        # print (normal.shape,"normal.shape")
        normal = self.deconv4(normal)
        # print (normal.shape,"normal.shape")
        normal = torch.nn.functional.normalize(normal, 2, 1)
        # print (normal.shape,"normal.shape")
        return normal

class top_down(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(top_down, self).__init__()
        c_in = 32  ## tejas
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # print("aaya")
        img   = x[0]
        # print (img.shape,"img.shape")
        # img_split = torch.split(img, 3, 1)
        img = img.permute(0,2,1,3)
        # if len(x) > 1: # Have lighting
        #     light = x[1]
        #     light_split = torch.split(light, 3, 1)

        # feats = []
        # for i in range(len(img_split)):
        #     net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
        #     feat, shape = self.extractor(net_in)
        #     feats.append(feat)
        feat,shape = self.extractor(img)
        # print (shape,"shape")
        # print (feat.shape,"feat.shape")
        # print(feat.shape,"feat.shape")
        # print(len(feats))  
        # print (len(img_split),"len(img_split)")   
        # print (feats[0].shape,"feats[0].shape")
        # print (len(feats),"len(feats)")

        # if self.fuse_type == 'mean':
        #     # feat_fused = torch.stack(feats, 1).mean(1)
        #     feats = feat.view(img.shape[0],-1)
        #     feat_fused = feats.mean(0)
        # elif self.fuse_type == 'max':
        #     # feat_fused, _ = torch.stack(feats, 1).max(1)
        #     feats = feat.view(img.shape[0],-1)
        #     feat_fused, _ = feats.max(0)
        # print (feat_fused.shape,"feat_fused.shape")
        normal = self.regressor(feat, shape)
        # print (normal.shape,"normal.shape")
        return normal


### Train PS-FCN on both synthetic datasets using 32 images-light pairs
# CUDA_VISIBLE_DEVICES=0 python main.py --concat_data --in_img_num 32 --model=top_down