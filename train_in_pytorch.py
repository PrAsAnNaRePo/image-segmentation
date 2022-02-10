import os
import numpy as np
import torch.utils.data
from PIL import Image
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.functional import relu, sigmoid
from torchvision.transforms import transforms, Compose
from torch.utils.data import Dataset, DataLoader


IMG_PATH = 'jpeg_images/IMAGES'
MASK_PATH = 'jpeg_masks/MASKS'
IMG_SIZE = (180, 180)
EPOCHS = 40
BATCH_SIZE = 20
LR = 0.001

class CustomDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.images = os.listdir(img_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        i_p = f'{self.img_path}/{self.images[item]}'
        m_p = f'{self.mask_path}/{self.images[item]}'.replace('img', 'seg')
        image = np.array(Image.open(i_p).convert('RGB'))
        mask = np.array(Image.open(m_p).convert('L'), dtype='float32')
        mask[mask == 255.0] = 1.0

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


trans = Compose([
    transforms.ToTensor(),
    transforms.Resize(size=IMG_SIZE),
    transforms.RandomAutocontrast(),
])

data = CustomDataset(IMG_PATH, MASK_PATH, transform=trans)

train_set, test_set = torch.utils.data.random_split(data, [850, 150])

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)


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

model = Model().to('cuda')

loss = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=LR)

def accuarcy(loader, model, device='cuda', e=0):
    n_correct=0
    n_pixels = 0
    dice_score = 0
    model.eval()
    best_score = 0.868

    with torch.no_grad():

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = torch.sigmoid(model(x))
            pred = (pred > 0.5).float()
            n_correct += (pred == y).sum()
            n_pixels += torch.numel(pred)
            dice_score += (1 * (pred * y).sum()) / ((pred + y).sum() + 1e-8)

    print(f'Got {n_correct}/{n_pixels} with acc {n_correct/n_pixels*100:.2f}')
    print(f'Dice Score : {dice_score/len(loader)}')
    if dice_score/len(loader) >= best_score:
        print('====> saving checkpoint..')
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, f"{e}-{dice_score/len(loader):.2f}model1_checkpoint.pth.tar")
    model.train()


for e in range(EPOCHS):
    for i, (im, m) in enumerate(train_loader):
        im = im.to('cuda')
        m = m.to('cuda')
        y_pred = sigmoid(model(im))
        l = loss(y_pred, m)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(f'epochs {e+1} with loss {l:.2f}')
    accuarcy(test_loader, model, e=e)


