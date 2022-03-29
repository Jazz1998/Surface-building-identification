import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import albumentations as A
import torch
import torch.nn as nn
import torch.utils.data as D
import torchvision
from rle_img import rle_encode, rle_decode
from tcdatasets import TianChiDataset
from torchvision import transforms as T
import segmentation_models_pytorch as smp

EPOCHES = 20
BATCH_SIZE = 8
IMAGE_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_root = '../user_data/model_data'
print(DEVICE)

def get_model():
    '''
    model = torchvision.models.segmentation.fcn_resnet50(True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    '''
    model = smp.Unet(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights='imagenet',  # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )
    return model

@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc

bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.8 * bce + 0.2 * dice

def main():
    # --------------------------加载数据及数据增强----------------------------
    train_mask = pd.read_csv('../tcdata/train_mask.csv', sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: '../tcdata/train/' + x)
    mask = rle_decode(train_mask['mask'].iloc[0])
    print(rle_encode(mask) == train_mask['mask'].iloc[0])
    # 数据增强
    trfm = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.VerticalFlip(p=0.5),  # 垂直翻转
        A.RandomRotate90(),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness(),
            A.ColorJitter(brightness=0.07, contrast=0.07,
                          saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3),
    ])
    dataset = TianChiDataset(
        train_mask['name'].values,
        train_mask['mask'].fillna('').values,
        trfm, False
    )
    valid_idx, train_idx = [], []
    for i in range(len(dataset)):
        if i % 7 == 0:
            valid_idx.append(i)
        #     else:
        elif i % 7 == 1:
            train_idx.append(i)
    train_ds = D.Subset(dataset, train_idx)
    valid_ds = D.Subset(dataset, valid_idx)
    # define training and validation data loaders
    loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    vloader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ----------------------------加载模型及优化器------------------------------------
    model = get_model()
    model.to(DEVICE)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    best_loss = 10
    train_loss = []

    # ----------------------------训练-----------------------------------
    print('start')
    for epoch in range(EPOCHES):
        losses = []
        model.train()
        for image, target in loader:
            image, target = image.to(DEVICE), target.float().to(DEVICE)
            optimizer.zero_grad()
            # output = model(image)['out']
            output = model(image)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(loss.item())
        print(f"train epoch {epoch}/{EPOCHES} loss: {np.array(losses).mean():.4f}")

        vloss = validation(model, vloader, loss_fn)
        # print(raw_line.format(epoch, np.array(losses).mean(), vloss,(time.time() - start_time) / 60 ** 1))
        print(f"valid epoch {epoch}/{EPOCHES} loss: {vloss:.4f}")
        train_loss.append(np.array(losses).mean())
        if vloss < best_loss:
            best_loss = vloss
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'train_loss': train_loss}
            # torch.save(state, '../user_data/model_data/model_best.pth')
            torch.save(model, '{0}/model.pth'.format(model_root))

    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(train_loss, label="loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('../user_data/tmp/loss.png')

# --------------------------------验证-----------------------------------
def test():
    trfm = T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
    ])
    subm = []
    model = torch.load(os.path.join(model_root, 'model.pth'))
    model.to(DEVICE)
    model.eval()
    test_mask = pd.read_csv('../tcdata/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: '../tcdata/test_a/' + x)

    for idx, name in enumerate(test_mask['name'].iloc[:]):
        image = cv2.imread(name)
        image = trfm(image)
        with torch.no_grad():
            image = image.to(DEVICE)[None]
            # 测试集数据扩增（Test Time Augmentation，简称TTA）
            score1 = model(image).cpu().numpy()

            score2 = model(torch.flip(image, [0, 3]))
            score2 = torch.flip(score2, [3, 0])[0][0].cpu().numpy()

            score3 = model(torch.flip(image, [0, 2]))
            score3 = torch.flip(score3, [2, 0])[0][0].cpu().numpy()

            score = (score1 + score2 + score3) / 3.0
            score_sigmoid = score.sigmoid().cpu().numpy()
            score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))
            # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
    subm = pd.DataFrame(subm)
    subm.to_csv('../prediction_result/result.csv', index=None, header=None, sep='\t')
    # plt.imsave('./output.png',rle_decode(subm[1].fillna('').iloc[0]), cmap='gray')

if __name__ == '__main__':
    main()
    test()


