import torch
import os
import torchvision
import matplotlib.pylab as plt

from torch.utils.data import Dataset, DataLoader

class SceneryDataset(Dataset):
    def __init__(self, image_dir):
        self.features = []
        for file in os.listdir(image_dir):
            if file.endswith(".png"):
                self.features.append(torchvision.io.read_image(os.path.join(image_dir, file)))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512),
            torchvision.transforms.RandomCrop(256),
            torchvision.transforms.RandomHorizontalFlip()
        ])

        # # gpu
        # self.features = [feature.cuda() for feature in self.features]

        print("read "+str(len(self.features))+" images from "+image_dir)

    def __getitem__(self, idx):
        return self.transform(self.features[idx]) / 255


    def __len__(self):
        return len(self.features)


def load_data_scenery(batch_size):
    """返回两个图像集合的迭代器"""
    A_iter = DataLoader(
        SceneryDataset("./train/trainA"), batch_size, shuffle=True)
    B_iter = DataLoader(
        SceneryDataset("./train/trainB"), batch_size, shuffle=True)
    return A_iter, B_iter


def rand_crop(feature, height, width):
    resize = torchvision.transforms.Resize(256)
    feature = resize(feature)
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    return feature


if __name__ == "__main__":
    A_iter, B_iter = load_data_scenery(8)
    for item in B_iter:
        for i in range(item.shape[0]):
            plt.imshow(item[i].permute(1, 2, 0))
            plt.show()
