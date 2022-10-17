import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
import pdb
import os
# import faiss
# import h5py
import numpy as np
from model.position_encoding import build_position_encoding
from model.transformer import build_transformer
from model.pmm import PMMs
import torchvision
from util.util import mask_from_tensor
from model.decoder import predBlock1


class UGTRNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, T=16, K=50, classes=2, zoom_factor=8, use_ppm=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True,
                 dataset_name='', args=None):
        super(UGTRNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes == 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion
        self.args = args
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.hidden_dim = 512
        self.input_proj = nn.Sequential(nn.Conv2d(2048, self.hidden_dim, kernel_size=1, bias=False),
                                        BatchNorm(self.hidden_dim), nn.ReLU(inplace=True), nn.Dropout2d(p=dropout))
        self.pmm = PMMs(self.hidden_dim, T)
        self.position_encoding = build_position_encoding(self.hidden_dim, 'v2')
        self.transformer = build_transformer(self.hidden_dim, dropout, nheads=8, dim_feedforward=2048, enc_layers=3,
                                             dec_layers=3, pre_norm=True, decoder_input_len=45 * 45, num_queries=T)
        self.conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False)

        self.mean_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.std_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)

        self.pred = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        # self.pred3 = nn.Conv2d(self.hidden_dim * 2, 3, kernel_size=1)
        # self.decoder1 = predBlock1(self.hidden_dim, self.hidden_dim)
        # self.decoder2 = predBlock1(self.hidden_dim, self.hidden_dim // 2)
        # self.decoder3 = predBlock1(self.hidden_dim // 2, self.hidden_dim // 4)
        # self.decoder4 = predBlock1(self.hidden_dim // 4, 3)
        self.kl_loss = nn.KLDivLoss(reduction='mean')
        self.K = K
        self.m_items = F.normalize(torch.rand((16, self.hidden_dim), dtype=torch.float),
                                   dim=1).cuda()  # Initialize the memory items
        kernel = torch.ones((7, 7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # kernel = np.repeat(kernel, 1, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()  # type: Variable
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z

    def forward(self, x, y=None):
        x_size = x.size()

        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # step1. backbone feature
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # step2.
        x = self.input_proj(x)

        residual = self.conv(x)
        mean = self.mean_conv(x)
        std = self.std_conv(x)

        prob_x = self.reparameterize(mean, std, 1)
        prob_out2 = self.reparameterize(mean, std, self.K)
        prob_out2 = torch.sigmoid(prob_out2)

        # uncertainty
        uncertainty = prob_out2.var(dim=1, keepdim=True).detach()
        if self.training:
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        residual *= (1 - uncertainty)

        ############## mask ##################### if not use mask , remove it !
        if self.training:
            rand_mask = uncertainty < torch.Tensor(np.random.random(uncertainty.size())).to(uncertainty.device)
            residual *= rand_mask.to(torch.float32)
        #########################################

        mean3 = prob_out2.mean(dim=1, keepdim=True)
        std3 = prob_out2.var(dim=1, keepdim=True)

        # step3. position encoding and gmm encoding
        
        x, mask = mask_from_tensor(x)
        print(x.shape)
        position_encoding = self.position_encoding(x, mask).to(x.device)
        # x = x + position_encoding
        # print('x',x.shape)
        x, z_, P_b, P_f = self.pmm(x)  # out z:[1, 2025, 16]
        # print('x_z',len(x),x[0].shape,z_.shape)
        x = torch.stack(x, dim=3).squeeze(-1)  # out x :[1, 512, 1, 16])
        position_encoding = torch.bmm(position_encoding.flatten(2), z_).unsqueeze(2)

        # transformer
        # ''' #if not use transformer , remove it !
        x, self.m_items, gathering_loss, spreading_loss, vis_att = self.transformer(x, residual, position_encoding,
                                                                                    self.m_items, self.training)
        vis_att = vis_att.transpose(1, 2)
        vis_att_size = vis_att.shape
        vis_att = vis_att.view(vis_att_size[0], vis_att_size[1], int(vis_att_size[2] ** 0.5),
                               int(vis_att_size[2] ** 0.5))
        # '''
        ################# just UQN: mask transformer ,this not use transformer
        # x = residual
        # vis_att = None
        # t_loss = 0
        # print("No transformer")
        #################

        # if x.size(1) == 1:

        x = self.pred(x)  # out:[1, 1, 60, 60] # ori

        #

        # elif x.size(1) == 3:
        #     # x = self.decoder1(x) #out:[1, 1, 60, 60] # ori
        #     # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #     # x = self.decoder2(x)
        #     # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #     # x = self.decoder3(x)
        #     # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        #     # x = self.decoder4(x)
        #     x = self.pred3(x)
        if self.zoom_factor != 1:
            prob_x = F.interpolate(prob_x, size=(h, w), mode='bilinear', align_corners=True)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            mean3 = F.interpolate(mean3, size=(h, w), mode='bilinear', align_corners=True)
            std3 = F.interpolate(std3, size=(h, w), mode='bilinear', align_corners=True)
            # uncertainty = F.interpolate(uncertainty, size=(h, w), mode='bilinear', align_corners=True)

        assert torch.sum(torch.isnan(prob_x)).item() == 0
        assert torch.sum(torch.isinf(prob_x)).item() == 0
        assert torch.sum(torch.isnan(x)).item() == 0
        assert torch.sum(torch.isinf(x)).item() == 0

        if self.training:
            y_flatten = y.view(y.size(0), y.size(1), -1)
            prob_x_flatten = prob_x.view(prob_x.size(0), prob_x.size(1), -1)

            BCE_loss = self.criterion(x, y)
            prob_loss = self.criterion(prob_x, y)
            KL_loss = self.kl_loss(F.log_softmax(prob_x_flatten, dim=-1), F.softmax(y_flatten, dim=-1)).sum()
            # main_loss = BCE_loss + 0.5 * prob_loss + 0.5 * t_loss + 0.5 * KL_loss
            # main_loss = BCE_loss + 0.5 * prob_loss + 0.5 * KL_loss

            # print(BCE_loss.item(),prob_loss.item(),gathering_loss.item(),spreading_loss.item(),KL_loss.item())
            main_loss = BCE_loss + 0.5 * prob_loss + 0.5 * (gathering_loss+spreading_loss) + 0.5 * KL_loss
            # print(BCE_loss.item(),prob_loss.item(),t_loss.item(),KL_loss.item())
            return torch.sigmoid(x), main_loss
        else:
            return torch.sigmoid(x), ((std3 - std3.min()) / (std3.max() - std3.min())), mean3, vis_att  # , uncertainty

