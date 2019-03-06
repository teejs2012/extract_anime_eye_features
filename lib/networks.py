import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools

class anime_full_encoder(nn.Module):
    def __init__(self,input_nc,first_filter=4,downsample=7, latent_dim=2048,output_dim=64,use_batch=False):
        super(anime_full_encoder,self).__init__()
        if use_batch:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        downconv_model=[nn.Conv2d(input_nc,first_filter,kernel_size=7,stride=1,padding=3),
                     norm_layer(first_filter),
                    nn.ReLU(True)]
        mult=1
        for i in range(downsample):
            downconv_model+=[nn.Conv2d(first_filter*mult,first_filter*mult*2,kernel_size=3,stride=2,padding=1),
                    norm_layer(first_filter*mult*2),
                    nn.ReLU(True)]
            mult = mult*2

        upconv_model = []
        for i in range(downsample):      
            upconv_model += [nn.ConvTranspose2d(first_filter * mult, first_filter * mult//2,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1),
                          norm_layer(first_filter * mult//2),
                          nn.ReLU(True)]
            mult = mult//2
        upconv_model += [nn.ReflectionPad2d(3),
                        nn.Conv2d(first_filter, 3, kernel_size=7, padding=0),
                        nn.Tanh()]
        self.upconv = nn.Sequential(*upconv_model)
        self.downconv = nn.Sequential(*downconv_model)
        self.fc1 = nn.Linear(latent_dim, output_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)
        self.downconv = init_weights(self.downconv)
        self.upconv = init_weights(self.upconv)
        self.fc1 = init_weights(self.fc1)
        self.fc2 = init_weights(self.fc2)

    def forward(self,input):
        latent = self.downconv(input)
        recon = self.upconv(latent)
        result = latent.view(-1)
        result_left = self.fc1(result)
        result_right = self.fc2(result)
        return recon,result_left,result_right

class anime_eye_encoder(nn.Module):
    def __init__(self,input_nc,first_filter=32,downsample=4, latent_dim=2048,output_dim=64,use_batch=False):
        super(anime_eye_encoder,self).__init__()
        if use_batch:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=False)
        else:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        downconv_model=[nn.Conv2d(input_nc,first_filter,kernel_size=7,stride=1,padding=3),
             norm_layer(first_filter),
            nn.ReLU(True)]
        mult=1
        for i in range(downsample):
            downconv_model+=[nn.Conv2d(first_filter*mult,first_filter*mult*2,kernel_size=3,stride=2,padding=1),
                    norm_layer(first_filter*mult*2),
                    nn.ReLU(True)]
            mult = mult*2
        upconv_model = []
        for i in range(downsample):      
            upconv_model += [nn.ConvTranspose2d(first_filter * mult, first_filter * mult//2,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1),
                          norm_layer(first_filter * mult//2,),
                          nn.ReLU(True)]
            mult = mult//2
        upconv_model += [nn.ReflectionPad2d(3),
                        nn.Conv2d(first_filter, 3, kernel_size=7, padding=0),
                        nn.Tanh()]
        self.upconv = nn.Sequential(*upconv_model)
        self.downconv = nn.Sequential(*downconv_model)
        self.fc1 = nn.Linear(latent_dim, output_dim)
        self.fc2 = nn.Linear(output_dim,latent_dim)
        self.upconv = init_weights(self.upconv)
        self.downconv = init_weights(self.downconv)
        self.fc1 = init_weights(self.fc1)
    def forward(self,input):
        latent = self.downconv(input)
        latent_shape = latent.size()
        latent = latent.view(-1)
        latent = self.fc1(latent)
        recon = self.fc2(latent)
        recon = recon.view(latent_shape)
        recon = self.upconv(recon)
        return recon,latent

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net