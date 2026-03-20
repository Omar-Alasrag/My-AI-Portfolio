import os
import torch as pt
from torch.nn import BCELoss
from torch.optim import Adam
import torchvision as tv
import torchvision.utils as vutils
import torchvision.transforms as T


transforms = T.Compose([T.Resize((64, 64)), T.ToTensor()])
dset = tv.datasets.ImageFolder(r"cifar10/train", transforms)
dataloader = pt.utils.data.DataLoader(dset, batch_size=32, shuffle=True)

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


class G(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = pt.nn.Sequential(
            pt.nn.ConvTranspose2d(100, 512, (4, 4), 1, 0),
            pt.nn.BatchNorm2d(512),
            pt.nn.ReLU(True),
            pt.nn.ConvTranspose2d(512, 256, (4, 4), 2, 1),
            pt.nn.BatchNorm2d(256),
            pt.nn.ReLU(True),
            pt.nn.ConvTranspose2d(256, 128, (4, 4), 2, 1),
            pt.nn.BatchNorm2d(128),
            pt.nn.ReLU(True),
            pt.nn.ConvTranspose2d(128, 64, (4, 4), 2, 1),
            pt.nn.BatchNorm2d(64),
            pt.nn.ReLU(True),
            pt.nn.ConvTranspose2d(64, 3, (4, 4), 2, 1),
            pt.nn.Tanh(),
        )

    def forward(self, x):
        x = self.main(x)
        return x


class D(pt.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = pt.nn.Sequential(
            pt.nn.Conv2d(3, 64, 4, 2, 1),
            pt.nn.LeakyReLU(0.2, True),  # Standard for Discriminators
            pt.nn.Conv2d(64, 128, 4, 2, 1),
            pt.nn.BatchNorm2d(128),
            pt.nn.LeakyReLU(0.2, True),
            pt.nn.Conv2d(128, 256, 4, 2, 1),
            pt.nn.BatchNorm2d(256),
            pt.nn.LeakyReLU(0.2, True),
            pt.nn.Conv2d(256, 512, 4, 2, 1),
            pt.nn.BatchNorm2d(512),
            pt.nn.LeakyReLU(0.2, True),
            pt.nn.Conv2d(512, 1, 4, 1, 0),
            pt.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1)


gnn: pt.nn.Module = G().to(device)
dnn: pt.nn.Module = D().to(device)
optG = Adam(gnn.parameters(), lr=0.0002, betas=(0.5, 0.999))
optD = Adam(dnn.parameters(), lr=0.0002, betas=(0.5, 0.999))


criterien = BCELoss()
for e in range(25):
    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.shape[0]
        optD.zero_grad()

        noise = pt.randn((batch_size, 100, 1, 1)).to(device)
        fake = gnn(noise)
        pred_fake = dnn(fake.detach())
        loss_fake = criterien(pred_fake, pt.zeros(real.shape[0]).to(device))

        pred_real = dnn(real)
        loss_real = criterien(pred_real, pt.ones(real.shape[0]).to(device))

        lossD = loss_real + loss_fake
        lossD.backward()
        optD.step()

        optG.zero_grad()
        Dout = dnn(fake)
        lossG = criterien(Dout, pt.ones(pred_real.shape[0]).to(device))
        lossG.backward()
        optG.step()

        if i % 100 == 0:
            print(
                f"Epoch [{e}/{25}] Batch {i} Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}"
            )

            os.makedirs("data", exist_ok=True)
            fake = gnn(noise)
            vutils.save_image(fake, f"data\\fake e={e}, i={i//100}.png", normalize=True)
