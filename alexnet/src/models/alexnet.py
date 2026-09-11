import torch
import torch.nn as nn


class ModernAlexNet(nn.Module):
    """
    Modernized AlexNet Neural Network Architecture.
    
    Improvements over original 2012 AlexNet:
    - Batch Normalization option for faster training & stable gradients
    - AdaptiveAvgPool2d to support dynamic input image sizes (e.g. 32x32 up to 224x224+)
    - Configurable dropout rate
    - Flexible input channels (RGB or grayscale) and variable output classes
    - Kaiming Normal weight initialization
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        use_batch_norm: bool = True,
        dropout: float = 0.5,
    ):
        super(ModernAlexNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout

        features_layers = []

        def conv_block(in_c, out_c, kernel_size, stride, padding):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            return layers

        # Layer 1
        features_layers.extend(conv_block(in_channels, 64, kernel_size=11, stride=4, padding=2))
        features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # Layer 2
        features_layers.extend(conv_block(64, 192, kernel_size=5, stride=1, padding=2))
        features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # Layer 3
        features_layers.extend(conv_block(192, 384, kernel_size=3, stride=1, padding=1))

        # Layer 4
        features_layers.extend(conv_block(384, 256, kernel_size=3, stride=1, padding=1))

        # Layer 5
        features_layers.extend(conv_block(256, 256, kernel_size=3, stride=1, padding=1))
        features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.features = nn.Sequential(*features_layers)

        # Adaptive pool guarantees 6x6 spatial dimensions regardless of input image resolution
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        classifier_layers = [
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ]
        self.classifier = nn.Sequential(*classifier_layers)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
