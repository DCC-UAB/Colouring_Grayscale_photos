from torch import nn

class Colorization_ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),#128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),#64
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),#32
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 2, 4, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x
