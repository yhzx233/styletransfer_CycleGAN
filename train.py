import torch
from torch import nn
import utils
import data
import matplotlib.pylab as plt
import random


class ImgBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, img):
        if (len(self.data) < self.max_size):
            self.data.append(img)
            return img
        else:
            if (random.uniform(0, 1) > 0.5):
                return img
            else:
                i = random.randint(0, self.max_size-1)
                res = self.data[i]
                self.data[i] = img
                return res


def train(gen1, gen2, disc1, disc2, X_iter, Y_iter,
          num_epoch=200,
          gan_weight=1, cycle_weight=10, identity_weight=2,
          learning_rate=0.0002, weight_decay=0,
          device=utils.try_gpu()):
    gen1 = gen1.to(device)
    gen2 = gen2.to(device)
    disc1 = disc1.to(device)
    disc2 = disc2.to(device)
    gen1buffer = ImgBuffer(50)
    gen2buffer = ImgBuffer(50)
    gen1_optimizer = torch.optim.Adam(
        gen1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    gen2_optimizer = torch.optim.Adam(
        gen2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    disc1_optimizer = torch.optim.Adam(
        disc1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    disc2_optimizer = torch.optim.Adam(
        disc2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    target_real = torch.tensor([[1.]], requires_grad=False).to(device)
    target_fake = torch.tensor([[0.]], requires_grad=False).to(device)
    for epoch in range(num_epoch):
        for i, (X, Y) in enumerate(zip(X_iter, Y_iter)):
            X = X.to(device)
            Y = Y.to(device)
            mseloss = torch.nn.MSELoss()
            for k in range(5):
                loss1 = (mseloss(disc1(X), target_real) +
                         mseloss(disc1(gen2buffer.push_and_pop(gen2(Y).detach())), target_fake))*0.5
                # loss1 = loss1.float()
                disc1_optimizer.zero_grad()
                loss1.backward()
                disc1_optimizer.step()

                loss2 = (mseloss(disc2(Y), target_real) +
                         mseloss(disc2(gen1buffer.push_and_pop(gen1(X).detach())), target_fake))*0.5
                disc2_optimizer.zero_grad()
                loss2.backward()
                disc2_optimizer.step()

            L1loss = torch.nn.L1Loss()

            gen1_identity_loss = L1loss(Y, gen1(Y))
            gen1_gan_loss = mseloss(disc2(gen1(X)), target_real)
            gen1_cycle_loss = L1loss(gen2(gen1(X)), X)
            gen1_loss = gen1_identity_loss*identity_weight + \
                gen1_cycle_loss*cycle_weight+gen1_gan_loss*gan_weight
            gen1_optimizer.zero_grad()
            gen1_loss.backward()
            gen1_optimizer.step()

            gen2_identity_loss = L1loss(X, gen2(X))
            gen2_gan_loss = mseloss(disc1(gen2(Y)), target_real)
            gen2_cycle_loss = L1loss(gen1(gen2(Y)), Y)
            gen2_loss = gen2_identity_loss*identity_weight + \
                gen2_cycle_loss*cycle_weight+gen2_gan_loss*gan_weight
            gen2_optimizer.zero_grad()
            gen2_loss.backward()
            gen2_optimizer.step()

            print(
                f"epoch = epoch {epoch}, disc1_loss = {loss1}, disc2_loss = {loss2}, gen1_loss = {gen1_loss}, gen2_loss = {gen2_loss}")

        if (epoch+1) % 5 == 0:
            # X0 = X[0].detach().cpu().permute(1, 2, 0)
            # Y0 = gen1(X)[0].detach().cpu().permute(1, 2, 0)
            # fig, axs = plt.subplots(1, 2)
            # X0 = X0*0.5+0.5
            # Y0 = Y0*0.5+0.5
            # axs[0].imshow(X0)
            # axs[1].imshow(Y0)
            # plt.show()
            torch.save(gen1.state_dict(), "gen1.params")
            torch.save(gen2.state_dict(), "gen2.params")
            torch.save(disc1.state_dict(), "disc1.params")
            torch.save(disc2.state_dict(), "disc2.params")
import torch
from torch import nn
import utils
import data

from tqdm import tqdm

import matplotlib.pylab as plt

class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)
    
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def train(gen1 : nn.Module, gen2 : nn.Module, disc1 : nn.Module, disc2 : nn.Module, X_iter, Y_iter,
          num_epoch=114,
          adv_weight=1, cycle_weight=5, identity_weight=5,
          learning_rate=0.0002, weight_decay=0,
          device=utils.try_gpu()):
    gen1 = gen1.to(device)
    gen2 = gen2.to(device)
    disc1 = disc1.to(device)
    disc2 = disc2.to(device)
    gen1_optimizer = torch.optim.Adam(
        gen1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    gen2_optimizer = torch.optim.Adam(
        gen2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    disc1_optimizer = torch.optim.Adam(
        disc1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    disc2_optimizer = torch.optim.Adam(
        disc2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.5, 0.999))
    

    gan_criteria = GANLoss().to(device)
    cycle_criteria = nn.L1Loss()
    identity_criteria = nn.L1Loss()

    for epoch in range(num_epoch):
        for i, (X, Y) in tqdm(enumerate(zip(X_iter, Y_iter)), total=min(len(X_iter), len(Y_iter))):
            # X = data.rand_crop(X).to(device)
            # Y = data.rand_crop(Y).to(device)
            X = X.to(device)
            Y = Y.to(device)
            fake_Y = gen1(X)
            fake_X = gen2(Y)

            gan_loss = gan_criteria(disc1(fake_Y), True) + gan_criteria(disc2(fake_X), True)
            cycle_loss = cycle_criteria(gen2(fake_Y), X) + cycle_criteria(gen1(fake_X), Y)
            idnetity_loss = identity_criteria(gen1(Y), Y) + identity_criteria(gen2(X), X)

            gen1.zero_grad()
            gen2.zero_grad()
            (gan_loss*adv_weight + cycle_loss*cycle_weight + idnetity_loss*identity_weight).backward()
            gen1_optimizer.step()
            gen2_optimizer.step()

            fake_Y = gen1(X).detach()
            fake_X = gen2(Y).detach()

            real_gan_loss = gan_criteria(disc1(Y), True) + gan_criteria(disc2(X), True)
            fake_gan_loss = gan_criteria(disc1(fake_Y), False) + gan_criteria(disc2(fake_X), False)

            disc1.zero_grad()
            disc2.zero_grad()
            (real_gan_loss + fake_gan_loss).backward()
            disc1_optimizer.step()
            disc2_optimizer.step()

            tqdm.write(f"epoch {epoch}, batch {i}, gan loss {gan_loss}, cycle loss {cycle_loss}, identity loss {idnetity_loss}")
            tqdm.write(f"real gan loss {real_gan_loss}, fake gan loss {fake_gan_loss}")
            # if i % 10 == 0:
            #     print(f"epoch {epoch}, batch {i}, gan loss {gan_loss}, cycle loss {cycle_loss}, identity loss {idnetity_loss}")
            #     print(f"real gan loss {real_gan_loss}, fake gan loss {fake_gan_loss}")
        if epoch % 1 == 0:
            # 显示在一个图里
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(X[0].detach().cpu().permute(1, 2, 0))
            axs[0, 1].imshow(gen1(X).detach().cpu()[0].permute(1, 2, 0))
            axs[1, 0].imshow(Y[0].detach().cpu().permute(1, 2, 0))
            axs[1, 1].imshow(gen2(Y).detach().cpu()[0].permute(1, 2, 0))
            # 保存为图片
            fig.savefig(f"sample/epoch_{epoch}.png")

        if (epoch+1) % 10 == 0:
            torch.save(gen1.state_dict(), f"model/gen1_{epoch}.pth")
            torch.save(gen2.state_dict(), f"model/gen2_{epoch}.pth")
            torch.save(disc1.state_dict(), f"model/disc1_{epoch}.pth")
            torch.save(disc2.state_dict(), f"model/disc2_{epoch}.pth")