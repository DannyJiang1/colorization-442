import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(1, 64, kernel_size=3, stride=1, padding=1)  # Input grayscale (L channel)
        self.enc2 = self._conv_block(64, 128, kernel_size=4, stride=2, padding=1)  # Downsample
        self.enc3 = self._conv_block(128, 256, kernel_size=4, stride=2, padding=1)  # Downsample
        self.enc4 = self._conv_block(256, 512, kernel_size=4, stride=2, padding=1)  # Downsample

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  # Updated padding
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),  # Updated padding
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec4 = self._upconv_block(512 + 512, 256)  # Skip connection (enc4 + bottleneck)
        self.dec3 = self._upconv_block(256 + 256, 128)  # Skip connection (enc3)
        self.dec2 = self._upconv_block(128 + 128, 64)   # Skip connection (enc2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),  # Output 2 channels (ab channels)
            nn.Tanh(),  # Normalize output to [-1, 1]
        )

        # Upsample
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Convolutional block with Conv -> BatchNorm -> ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, out_channels):
        """Upsampling block with Transposed Conv -> BatchNorm -> ReLU."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        # print(enc1.size(), enc2.size(), enc3.size(), enc4.size(), bottleneck.size())

        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([bottleneck, enc4], dim=1))  # Combine bottleneck and enc4 features
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))        # Combine dec4 and enc3 features
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))        # Combine dec3 and enc2 features
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))        # Combine dec2 and enc1 features
        # Upsample to original resolution
        return dec1