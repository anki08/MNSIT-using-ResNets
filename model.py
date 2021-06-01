import torch

class CNNClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.master = torch.nn.Sequential(torch.nn.Conv2d(1, 16, kernel_size=7, padding=3, stride=2),
                                          torch.nn.ReLU(),
                                          torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        torch.nn.init.xavier_normal_(self.master[0].weight)

        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                          torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                                          torch.nn.ReLU())

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU())

        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.ReLU())

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.downsample1 = torch.nn.Sequential(torch.nn.Conv2d(16, 32, kernel_size=1),
                                               torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                                               torch.nn.ReLU())
        self.downsample2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=1),
                                               torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                                               torch.nn.ReLU())
        self.downsample3 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=1),
                                               torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                                               torch.nn.ReLU())

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 10),
        )

    def forward(self, x):
        # print(x.shape)
        # normalize image

        mu = torch.mean(torch.mean(x, dim=2), dim=2).unsqueeze(-1).unsqueeze(-1)
        sigma = torch.sqrt(torch.mean((x - mu) ** 2)) + 1e-8
        x -= mu
        x /= 4 * sigma

        # print("image", identity.shape)
        res1 = self.master(x)

        res2 = self.block1(res1)
        res2 = res2 + self.downsample1(res1)
        res2 = self.maxpool(res2)

        res3 = self.block2(res2)
        res3 = res3 + self.downsample2(res2)
        res3 = self.maxpool(res3)

        res4 = self.block3(res3)
        # print("4 ", res4.shape ,self.downsample3(res3).shape )
        res4 = res4 + self.downsample3(res3)

        res = self.maxpool(res4)
        # print("final shape : ", res.shape)
        res = res.mean(dim=[2, 3])
        res = self.classifier(res)
        return res

model_factory = {
    'cnn': CNNClassifier
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
