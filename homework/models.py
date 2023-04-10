import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        """
        
        self.weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=self.weights)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=2048, bias=True)
        # print(self.resnet)
        self.avg = nn.AdaptiveAvgPool1d(output_size=2048)
        self.mlp = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,6)
        )
        
        # raise NotImplementedError('CNNClassifier.__init__') 

    def forward(self, x):
        """
        Your code here
        """
        
        out1 = self.resnet(x)
        # print(out1.shape)
        avg = self.avg(out1)
        avg = (avg.squeeze()).squeeze()
        output = self.mlp(avg)
        
        return output
        
        # raise NotImplementedError('CNNClassifier.forward') 


class FCN_ST(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Single-Task FCN needs to output segmentation maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        in_ch=3
        out_ch=19
        dim=64
        
        #  128*259
        self.en1 = nn.Sequential(
            torch.nn.Conv2d(in_channels = in_ch,out_channels = dim,kernel_size=4,stride=2,padding=1)
        )
        
        #   64*128
        self.en2 = nn.Sequential(
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(dim,dim*2,kernel_size=4,stride=2,padding=1),
            torch.nn.BatchNorm2d(dim*2)
        )
        
        #   32*64
        self.en3 = nn.Sequential(
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(dim*2,dim*4,kernel_size=4,stride=2,padding=1),
            torch.nn.BatchNorm2d(dim*4)
        )
        
        #   16*32
        self.en4 = nn.Sequential(
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(dim*4,dim*8,kernel_size=4,stride=2,padding=1),
            torch.nn.BatchNorm2d(dim*8)
        )
        
        #   8*16
        self.en5 = nn.Sequential(
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(dim*8,dim*8,kernel_size=4,stride=2,padding=1),
            torch.nn.BatchNorm2d(dim*8)
        )
        
        #   4*8
        self.en6 = nn.Sequential(
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(dim*8,dim*8,kernel_size=4,stride=2,padding=1),
            torch.nn.BatchNorm2d(dim*8)
        )
        
        #   2*4
        self.en7 = nn.Sequential(
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(dim*8,dim*8,kernel_size=4,stride=2,padding=1),
            # torch.nn.BatchNorm2d(dim*8)
        )
        
        #   1*2
        
        # self.en8 = nn.Sequential(
        #     torch.nn.LeakyReLU(0.2, inplace=True),
        #     torch.nn.Conv2d(dim*8,dim*8,kernel_size=4,stride=2,padding=1),
        # )
        
        
        #   start decoder
        #   1*2
        self.de1 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 8, dim* 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(dim* 8),
            torch.nn.Dropout(p=0.5)
        )
        
        # 2 * 4
        self.de2 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 8 * 2, dim* 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(dim* 8),
            torch.nn.Dropout(p=0.5)
        )
        
        # 4 * 8
        self.de3 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 8 * 2, dim* 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(dim* 8),
            torch.nn.Dropout(p=0.5)
        )
        
        # 8 * 16
        self.de4 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 8 * 2, dim* 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(dim* 4),
            torch.nn.Dropout(p=0.5)
        )
        
        # 16 * 32
        self.de5 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 4 * 2, dim* 2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(dim* 2),
            torch.nn.Dropout(p=0.5)
        )
        
        # 32 * 64
        self.de6 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 2 * 2, dim , kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(dim),
            torch.nn.Dropout(p=0.5)
        )
        
        # 64 * 128
        self.de7 = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(dim* 2, out_ch, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh(),
            torch.nn.Sigmoid()
        )
        
        # 128 * 256
        
        # self.de8 = nn.Sequential(
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.ConvTranspose2d(dim* 2, out_ch, kernel_size=4, stride=2, padding=1),
        #     torch.nn.Tanh()
        # )
        
        # raise NotImplementedError('FCN_ST.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation.
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding CNNClassifier
              convolution
        """
        
        en1_out = self.en1(x)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        en6_out = self.en6(en5_out)
        en7_out = self.en7(en6_out)
        # en8_out = self.en8(en7_out)

        # Decoder
        de1_out = self.de1(en7_out)
        de1_cat = torch.cat([de1_out, en6_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en5_out], dim=1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en4_out], dim=1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en3_out], dim=1)
        de5_out = self.de5(de4_cat)
        de5_cat = torch.cat([de5_out, en2_out], dim=1)
        de6_out = self.de6(de5_cat)
        de6_cat = torch.cat([de6_out, en1_out], dim=1)
        de7_out = self.de7(de6_cat)
        # de7_cat = torch.cat([de7_out, en1_out], dim=1)
        # de8_out = self.de8(de7_cat)

        return de7_out
        
        # raise NotImplementedError('FCN_ST.forward')


class FCN_MT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The Multi-Task FCN needs to output both segmentation and depth maps at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        
        self.basenet = FCN_ST()
        self.depthnet = nn.Conv2d(19,1,kernel_size=3,padding=1)
        
        # raise NotImplementedError('FCN_MT.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,C,H,W)), C is the number of classes for segmentation
        @return: torch.Tensor((B,1,H,W)), 1 is one channel for depth estimation
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        
        segmentation = self.basenet(x)
        depth = self.depthnet(segmentation)
        
        return segmentation, depth
        # raise NotImplementedError('FCN_MT.forward')


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):

        """
        Your code here
        Hint: inputs (prediction scores), targets (ground-truth labels)
        Hint: Implement a Softmax-CrossEntropy loss for classification
        Hint: return loss, F.cross_entropy(inputs, targets)
        """
        truth = torch.exp(inputs[targets])
        sum = torch.sum(torch.exp(inputs),dim = 1,keepdims = True)
        output = -torch.log(truth/sum)
        
        return torch.sum(output)
        
        # raise NotImplementedError('SoftmaxCrossEntropyLoss.__init__')


model_factory = {
    'cnn': CNNClassifier,
    'fcn_st': FCN_ST,
    'fcn_mt': FCN_MT
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
