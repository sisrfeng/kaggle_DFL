import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


DATA_ROOT ='../input/bundesliga'


#Show Mask
img = plt.imread(DATA_ROOT + '/masks/bundesliga10.png')
print(img.shape)
#Put off RGB
plt.imshow(img[..., 0])
img2=img[..., 0]
print(img2.shape)
print(np.unique(img * 255))


np.unique(img * 255)


# labels
labels = ['bleachers','ball','field']


from albumentations.pytorch import ToTensorV2
import albumentations as A
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,ToGray,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

t1 = A.Compose([
    A.Resize(256,256),
    #ToGray(p=0.3),
    #ShiftScaleRotate(p=0.1),
    #HorizontalFlip(),
    #RandomBrightnessContrast(p=0.2),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


num_masks=[0,50,100]


#separate mask in channels

for i in range(3):

    mask = plt.imread(DATA_ROOT + '/masks/bundesliga3.png') * 255
    #Lo que sea igual a i lo pone en 255 lo que no lo pone en 0
    mask = np.where(mask == num_masks[i], 255, 0)
    #solo la capa Red
    mask = mask[:,:,0]
    #print(mask)
    plt.title(f'class: {i} {labels[i]}')
    plt.imshow(mask)
    plt.show()


#Generate list of images
images = []
masks = []

for root, dirs, files in os.walk(DATA_ROOT):
    for name in files:
        f = os.path.join(root, name)
        if 'images' in f:
            images.append(f)
        elif 'masks' in f:
            masks.append(f)
        else:
            break


len(images),len(masks)


#Generate Panda Dataframe Images and Masks
df = pd.DataFrame({'images': images, 'masks': masks})

df.sort_values(by='images',inplace=True)

df.reset_index(drop=True, inplace=True)

df.head(5)


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch.nn import functional as F


class Segmentation(Dataset):
    def __init__(self, data,transform = None):
        self.transforms = transform
        self.data = data
        #Separate list of images[:,0] and list of masks[:,1]
        self.image_arr = self.data.iloc[:,0]
        self.label_arr = self.data.iloc[:,1]
        self.data_len = len(self.data.index)


    def __getitem__(self, index):

        #convert image to numpy array
        img = cv2.cvtColor(cv2.imread(self.image_arr[index]), cv2.COLOR_BGR2RGB)

        img = np.asarray(img)

        #convert mask to numpy array
        mask = cv2.cvtColor(cv2.imread(self.label_arr[index]), cv2.COLOR_BGR2RGB)
        mask = np.asarray(mask)

        #*********************************************
        #*******Separate channel 3 options  *********
        #*******  YOU CAN CHANGE THE CHANNEL *********
        #***** bleachers= 0 ball= 50 or field=100 ********************
        #*********************************************

        # Select mask ==0 bleachers,  mask ==50 ball, mask ==100 field
        cls_mask_1 = np.where(mask == 0, 1, 0)[:,:,0]

        #Apply albumentations in image and mask
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=cls_mask_1)
            img = aug['image']
            mask = aug['mask']

        return img, mask

    def __len__(self):
        return self.data_len


def get_images(image_dir,transform = None,batch_size=1,shuffle=True,pin_memory=True):
    data = Segmentation(image_dir,transform = t1)

    #70% train 30 % test
    train_size = int(0.7 * data.__len__())
    test_size = data.__len__() - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    #Generate Dataloader
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return train_batch,test_batch


train_batch,test_batch = get_images(df,transform =t1,batch_size=5)


#Show images and Masks
for img,mask in train_batch:

    img1 = np.transpose(img[1,:,:,:],(1,2,0))
    print(img1.shape)
    print(mask.shape)
    mask1 = np.array(mask[1,:,:])
    print(mask1.shape)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(img1)
    ax[1].imshow(mask1)

    break


pip install torchsummary


class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)


# Two channels
class Unet_model(nn.Module):
    def __init__(  self                               ,
                 out_channels = 2                   ,
                 features     = [64, 128, 256, 512] ,
                ):
        super(Unet_model,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(3,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = Unet_model().to(DEVICE)


from torchsummary import summary
summary(model, (3, 256, 256))


LEARNING_RATE = 1e-4
num_epochs = 50


loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()


for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_batch),total=len(train_batch))
    for batch_idx, (data, targets) in loop:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


check_accuracy(test_batch, model)


#you can update these cells to see other images and masks
for x,y in train_batch:
    x = x.to(DEVICE)
    fig , ax =  plt.subplots(1, 3, figsize=(18, 18))
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
    img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
    preds1 = np.array(preds[0,:,:])
    mask1 = np.array(y[0,:,:])

    ax[0].set_title('Image')
    ax[1].set_title('Prediction')
    ax[2].set_title('Mask')

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")

    ax[0].imshow(img1)
    ax[1].imshow(preds1)
    ax[2].imshow(mask1)

    break
