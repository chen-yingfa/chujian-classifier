import timm
from torch import nn
from timm.models.layers import PatchEmbed, DropPath



class Vit(nn.Module):
    def __init__(
        self,
        model_name: str,
        img_size: tuple,
        num_classes: int,
        pretrained: bool = False,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SmallVit(nn.Module):
    def __init__(
        self,
        model_name: str,
        img_size: tuple,
        num_classes: int,
        pretrained: bool = False
    ):
        super().__init__()
        # Patch embedding using timm library
        self.patch_embed = timm