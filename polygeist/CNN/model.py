"""
Implementation of EFFNet.
"""
import torch
import torch.nn as nn
import timm.models.efficientnet as effnet


class PDNet(nn.Module):
    """
    Our implementation of EFFNet.
    This implementation is used to load other model files and our own trained models.
    """

    def __init__(self):
        """Define model structure"""
        super(PDNet, self).__init__()

        self.encoder = effnet.tf_efficientnet_b7_ns()
        """Encoder layer: EfficientNet_b7_ns"""

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        """Average pooling layer: AdaptiveAvgPool2d"""

        self.layers = nn.ModuleDict(
            {
                "dropout_0": nn.Dropout(0.3),
                "fc_0": nn.Linear(2560, 1024),
                "dropout_1": nn.Dropout(0.3),
                "fc_1": nn.Linear(1024, 1),
            }
        )
        """Additional layers: 2x(Dropout and Fully Connected)"""

        self.final = nn.Sequential((nn.Sigmoid()))
        """Final layer: Sequential(Sigmoid)"""

    def forward(self, x):
        """Forward inference"""
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.layers["dropout_0"](x)
        x = self.layers["fc_0"](x)
        x = self.layers["dropout_1"](x)
        x = self.layers["fc_1"](x)
        x = self.final(x)
        return x

    def save_state(self, path):
        """Save model state to file"""
        torch.save(
            {
                "eff": self.encoder.state_dict(),
                "layers": self.state_dict(),
            },
            path,
        )

    def apply_state(self, state_path):
        """Load model state from file"""
        state = torch.load(state_path)
        self.load_state_dict(state["layers"])
        self.encoder.load_state_dict(state["eff"])
