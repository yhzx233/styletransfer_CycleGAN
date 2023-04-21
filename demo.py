import os
import data
import train
import model
import utils
import torch
import torchvision
import matplotlib.pylab as plt


device = utils.try_gpu()
gen1, gen2 = model.Generator(), model.Generator()
gen1.eval()
# gen2.eval()
gen1.load_state_dict(torch.load("model/gen1_109.pth"))
# gen2.load_state_dict(torch.load("gen2.params"))

image_dir = "./test"

files = os.listdir(image_dir)
images = []


for file in files:
    img = torchvision.io.read_image(
        os.path.join(image_dir, file))
    img = img/255
    # print(img.shape)
    img = img.unsqueeze(0)
    plt.imshow(gen1(img).detach()[0].permute(1, 2, 0))
    plt.show()
