import torch.nn as nn
import torch.nn.functional as F
import torch
#import os


class UNet(nn.Module):
    def __init__(self, dim, n_filters, FL, init, drop, lmbda):
        super(UNet, self).__init__()

        # Define layers
       # self.conv_no_padding = nn.Conv2d(100, n_filters,3,padding=1)
        self.conv1 = nn.Conv2d(1, n_filters, FL, padding=FL // 2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, FL, padding=FL // 2)
        self.conv3 = nn.Conv2d(n_filters, n_filters * 2, FL, padding=FL // 2)
        self.conv4 = nn.Conv2d(n_filters * 2, n_filters * 2, FL, padding=FL // 2)
        self.conv5 = nn.Conv2d(n_filters * 2, n_filters * 4, FL, padding=FL // 2)
        self.conv6 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL // 2)
        self.conv7 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL // 2)
        self.conv8 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL // 2)
        self.conv9 = nn.Conv2d(
            n_filters * 8, n_filters * 2, FL, padding=FL // 2
        )  # Changed input channels
        self.conv10 = nn.Conv2d(n_filters * 2, n_filters * 2, FL, padding=FL // 2)
        self.conv11 = nn.Conv2d(
            n_filters * 4, n_filters, FL, padding=FL // 2
        )  # Changed input channels
        self.conv12 = nn.Conv2d(n_filters, n_filters, FL, padding=FL // 2)
        self.conv13 = nn.Conv2d(
            n_filters * 2, n_filters, FL, padding=FL // 2
        )  # Added to match concatenation
        self.conv14 = nn.Conv2d(
            n_filters, n_filters, FL, padding=FL // 2
        )  # Added to match concatenation
        self.final_conv = nn.Conv2d(n_filters, 1, 1, padding=0)

        # Define pooling layers
        self.maxpool = nn.MaxPool2d(2, 2)

        # Define upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Define dropout layer
        self.dropout = nn.Dropout(drop)

        # Initialization
        if init == "he_normal":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif init == "glorot_uniform":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            raise ValueError("Unknown initialization: " + str(init))

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        
        a1 = F.relu(self.conv2(a1))
        
        a1P = self.maxpool(a1)

        a2 = F.relu(self.conv3(a1P))
        a2 = F.relu(self.conv4(a2))
        #print(a2.shape)
        a2P = self.maxpool(a2)

        a3 = F.relu(self.conv5(a2P))
        a3 = F.relu(self.conv6(a3))
        #print(a3.shape)
        a3P = self.maxpool(a3)

        u = F.relu(self.conv7(a3P))
        u = F.relu(self.conv8(u))

        u = self.upsample(u)
        
        u = torch.cat((a3, u), dim=1)
        u = self.dropout(u)
        u = F.relu(self.conv9(u))
        u = F.relu(self.conv10(u))

        u = self.upsample(u)
        u = torch.cat((a2, u), dim=1)
        u = self.dropout(u)
        u = F.relu(self.conv11(u))
        u = F.relu(self.conv12(u))

        u = self.upsample(u)
        
        #u = torch.nn.functional.pad(u, pad=(1,1))
        #4 = self.conv_no_padding(a1)
        #print(a4.shape)
        #print(u.shape)
        u = torch.cat((a1, u), dim=1)
        u = self.dropout(u)
        u = F.relu(self.conv13(u))
        u = F.relu(self.conv14(u))

        u = self.final_conv(u)
        return torch.sigmoid(u)
