import torch as tr

class Net(tr.nn.Module):
    def __init__(self, nlayer):
        super().__init__()
        self.lin = tr.nn.ModuleList(
            [tr.nn.Linear(nlayer[i], nlayer[i+1])
                for i in range(len(nlayer)-2)])
        self.lru = tr.nn.LeakyReLU()
        self.readout = tr.zeros(nlayer[-2], nlayer[-1])

    def get_loss(self, x, y):
        for lin in self.lin:
            x = self.lru(lin(x))
        self.readout = tr.linalg.lstsq(x, y).solution
        pred = tr.matmul(x, self.readout)
        loss = tr.sum((pred - y)**2)
        return loss

nlayer = [1, 8, 8, 1]
net = Net(nlayer)

batch_size = 8
x = tr.randn(batch_size, 1)
y = tr.sin(x)

opt = tr.optim.SGD(net.parameters(), lr=0.0001)

curve = []
for update in range(1000):
    opt.zero_grad()
    loss = net.get_loss(x, y)
    loss.backward()
    opt.step()
    curve.append(loss.item())
    print(update, curve[-1])

import matplotlib.pyplot as pt
pt.plot(curve)
pt.show()

