import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        # input: [batch_size, 1, 32, 32]
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        # output: [batch_size, 32, 16, 16]

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        # output: [batch_size, 64, 8, 8]

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        # output: [batch_size, 128, 4, 4]

        self.flatten = nn.Flatten()
        # output: [batch_size, 2048]

        self.fc = nn.Linear(2048, 2)
        # output: [batch_size, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        # input: [batch_size, 2]
        self.fc = nn.Linear(2, 2048)
        # output: [batch_size, 2048]

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 4, 4))
        # output: [batch_size, 128, 4, 4]

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        # output: [batch_size, 128, 8, 8]

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        # output: [batch_size, 64, 16, 16]

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        # output: [batch_size, 32, 32, 32]

        self.conv_final = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3, padding=1
        )
        # output: [batch_size, 1, 32, 32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.conv_final(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
