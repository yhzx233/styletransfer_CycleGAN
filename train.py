import torch
from torch import nn
import utils
import data


def train(gen1, gen2, disc1, disc2, X_iter, Y_iter,
          num_epoch=114,
          adv_weight=1, cycle_weight=3, identity_weight=10,
          learning_rate=0.00000002, weight_decay=0,
          device=utils.try_gpu()):
    gen1 = gen1.to(device)
    gen2 = gen2.to(device)
    disc1 = disc1.to(device)
    disc2 = disc2.to(device)
    gen1_optimizer = torch.optim.Adam(
        gen1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    gen2_optimizer = torch.optim.Adam(
        gen2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    disc1_optimizer = torch.optim.Adam(
        disc1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    disc2_optimizer = torch.optim.Adam(
        disc2.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epoch):
        for i, (X, Y) in enumerate(zip(X_iter, Y_iter)):
            # X = data.rand_crop(X).to(device)
            # Y = data.rand_crop(Y).to(device)
            X = X.to(device)
            Y = Y.to(device)
            loss1 = (torch.log(disc2(Y))+torch.log(1-disc2(gen1(X)))).sum()
            loss2 = (torch.log(disc1(X))+torch.log(1-disc1(gen2(Y)))).sum()
            loss3 = (torch.abs(gen2(gen1(X))-X) +
                     torch.abs(gen1(gen2(Y))-Y)).sum()
            loss = (loss1+loss2)*adv_weight+loss3*cycle_weight
            gen1_optimizer.zero_grad()
            gen2_optimizer.zero_grad()
            disc1_optimizer.zero_grad()
            disc2_optimizer.zero_grad()
            loss.backward()
            gen1_optimizer.step()
            gen2_optimizer.step()
            disc1_optimizer.step()
            disc2_optimizer.step()
            print(f"epoch = epoch {epoch}, loss = {loss}")
    torch.save(gen1.state_dict(), "gen1.params")
    torch.save(gen2.state_dict(), "gen2.params")
    torch.save(disc1.state_dict(), "disc1.params")
    torch.save(disc2.state_dict(), "disc2.params")
