import cv2
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.functional import relu, sigmoid
from torchvision.transforms import transforms, Compose

class Model(nn.Module):
    def __init__(self, n_inputchannel=3, n_outputchannel=256):
        super(Model, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=n_inputchannel, out_channels=n_outputchannel,
                               kernel_size=(3, 3),
                               padding=1, bias=False)
        self.bth1 = nn.BatchNorm2d(n_outputchannel)
        self.max1 = nn.MaxPool2d(kernel_size=(2, 2))  # 256x90x90

        self.Conv2 = nn.Conv2d(in_channels=n_outputchannel, out_channels=128,
                               kernel_size=(3, 3),
                               padding=1,
                               bias=False)
        self.bth2 = nn.BatchNorm2d(128)
        self.max2 = nn.MaxPool2d(kernel_size=(2, 2))  # 128x45x45

        self.Conv3 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=(1, 1),
                               padding=1,
                               bias=False)
        self.bth3 = nn.BatchNorm2d(64)
        self.max3 = nn.MaxPool2d(kernel_size=(3, 3))  # 64x15x15

        self.upsample1 = nn.ConvTranspose2d(in_channels=64,out_channels=256, kernel_size=(3,3))
        self.Conv4 = nn.Conv2d(256, 256, kernel_size=(1, 1), padding=1)
        self.bth4 = nn.BatchNorm2d(256)  # 256x19x19

        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5))
        self.Conv5 = nn.Conv2d(128, 128, kernel_size=(1, 1), padding=1)
        self.bth5 = nn.BatchNorm2d(128)  # 128x?x?

        self.upsample3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5))
        self.Conv6 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding=1)
        self.bth6 = nn.BatchNorm2d(64)  # 64x31x31

        self.upsample4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5))
        self.Conv7 = nn.Conv2d(32, 32, kernel_size=(1, 1), padding=1)
        self.bth7 = nn.BatchNorm2d(32)  # 32x37x37

        self.upsample5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(7, 7), stride=(3, 3))
        self.Conv8 = nn.Conv2d(16, 16, kernel_size=(1, 1), padding=1)
        self.bth8 = nn.BatchNorm2d(16)  # 16x117x117

        self.upsample6 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(11, 11), stride=(1, 1))
        self.Conv9 = nn.Conv2d(8, 8, kernel_size=(1, 1), padding=1)
        self.bth9 = nn.BatchNorm2d(8)  # 8x129x129

        self.upsample7 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(11, 11), stride=(1, 1))
        self.Conv10 = nn.Conv2d(4, 4, kernel_size=(1, 1), padding=1)
        self.bth10 = nn.BatchNorm2d(4)  # 4x141x141

        self.upsample8 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(11, 11), stride=(1, 1))
        self.Conv11 = nn.Conv2d(2, 2, kernel_size=(1, 1), padding=1)
        self.bth11 = nn.BatchNorm2d(2)  # 2x153x153

        self.upsample9 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(13, 13), stride=(1, 1))
        self.Conv12 = nn.Conv2d(2, 2, kernel_size=(1, 1), padding=1)
        self.bth12 = nn.BatchNorm2d(2)  # 2x167x167

        self.upsample10 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(12, 12), stride=(1, 1))
        self.Conv13 = nn.Conv2d(1, 1, kernel_size=(1, 1), padding=1)
        self.bth13 = nn.BatchNorm2d(1)  # 1x180x180

    def forward(self, x):
        x = self.Conv1(x)
        x = relu(self.bth1(x))
        x = self.max1(x)
        x = self.Conv2(x)
        x = relu(self.bth2(x))
        x = self.max2(x)

        x = self.Conv3(x)
        x = relu(self.bth3(x))
        x = self.max3(x)

        x = self.upsample1(x)
        x = self.Conv4(x)
        x = relu(self.bth4(x))

        x = self.upsample2(x)
        x = self.Conv5(x)
        x = relu(self.bth5(x))

        x = self.upsample3(x)
        x = self.Conv6(x)
        x = relu(self.bth6(x))

        x = self.upsample4(x)
        x = self.Conv7(x)
        x = relu(self.bth7(x))

        x = self.upsample5(x)
        x = self.Conv8(x)
        x = relu(self.bth8(x))

        x = self.upsample6(x)
        x = self.Conv9(x)
        x = relu(self.bth9(x))

        x = self.upsample7(x)
        x = self.Conv10(x)
        x = relu(self.bth10(x))

        x = self.upsample8(x)
        x = self.Conv11(x)
        x = relu(self.bth11(x))

        x = self.upsample9(x)
        x = self.Conv12(x)
        x = relu(self.bth12(x))

        x = self.upsample10(x)
        x = self.Conv13(x)
        x = relu(self.bth13(x))

        # x = sigmoid(x)

        return x

img = cv2.imread(r"D:\Spider\image-segmentation\jpeg_images\IMAGES\img_0032.jpeg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (180, 180))
img = img.reshape(1, 3, 180, 180)

input_tensor = torch.from_numpy(img)
input_tensor = torch.tensor(input_tensor, dtype=torch.float32, device='cuda')
model = Model().to('cuda')
checkpoint = '27-0.87model1_checkpoint.pth.tar'


model.load_state_dict(torch.load(checkpoint)['state_dict'])
optimizer = Adam(model.parameters(), lr=0.001)

pred = model(input_tensor)

pred = numpy.array(pred.cpu().detach().numpy()).reshape(180, 180, 1)

plt.imshow(pred)
plt.show()

# cv2.imshow('predicted', pred)
# cv2.waitKey()