import torch
import os
import torchvision
import matplotlib.pylab as plt


def read_data_scenery(image_dir):
    files = os.listdir(image_dir)
    images = []
    resize = torchvision.transforms.Resize(512)
    for file in files:
        feature = resize(torchvision.io.read_image(
            os.path.join(image_dir, file)))
        rect = torchvision.transforms.RandomCrop.get_params(
            feature, (256, 256))
        feature = torchvision.transforms.functional.crop(feature, *rect)
        images.append(feature)
    return images


class SceneryDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.features = read_data_scenery(image_dir)
        print("read "+str(len(self.features))+" images from "+image_dir)

    def __getitem__(self, idx):
        return self.features[idx].float()

    def __len__(self):
        return len(self.features)


def load_data_scenery(batch_size):
    """返回两个图像集合的迭代器"""
    A_iter = torch.utils.data.DataLoader(
        SceneryDataset("./train/trainA"), batch_size, shuffle=True)
    B_iter = torch.utils.data.DataLoader(
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
            plt.imshow(rand_crop(item[i], 200, 200).permute(1, 2, 0).int())
            plt.show()
