"""
用神经网络模拟微分方程,f(x)'=f(x),初始条件f(0) = 1
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self, NL, NN):
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.hidden_layer = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])
        self.output_layer = nn.Linear(NN, 1)

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layer):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out

    def act(self, x):
        return torch.tanh(x)
if __name__ == "__main__":
    x = torch.linspace(0, 2, 2000,requires_grad=True).unsqueeze(-1)
    y = torch.exp(x)
    net = Net(4, 20)
    lr = 1e-4
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(),lr)
    plt.ion()
    for i in range(10 ** 4):

        y_0 = net(torch.zeros(1))
        dx = torch.autograd.grad(net(x), x, grad_outputs=torch.ones_like(net(x)), create_graph=True)[0]
        optimizer.zero_grad()
        y_train = net(x)
        Mse1 = loss_fn(y_train,dx)
        Mse2 = loss_fn(y_0,torch.ones(1))
        loss = Mse1 + Mse2
        if i % 2000 == 0:
            plt.cla()
            plt.scatter(x.detach().numpy(),y.detach().numpy())
            plt.plot(x.detach().numpy(), y_train.detach().numpy(), c='red',lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
            print(f'times {i} - lr {lr} -  loss: {loss.item()} - y_0: {y_0}')
        loss.backward()
        optimizer.step()
    plt.ioff()
    plt.show()
    y_1 = net(torch.ones(1))
    print(f'y_1:{y_1}')
    y2 = net(x)
    plt.plot(x.detach().numpy(), y.detach().numpy(), c='red', label='True')
    plt.plot(x.detach().numpy(), y2.detach().numpy(), c='blue', label='Pred')
    plt.legend(loc='best')

    plt.show()