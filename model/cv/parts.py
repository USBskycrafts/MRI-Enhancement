import logging
import torch
import torch.nn as nn
from model.unet.unet_parts import *
from .loss import GramLoss, ReconstructionLoss, ElementLoss


class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.input_dim = input_channels

        self.inc = DoubleConv(self.input_dim, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.logger = logging.getLogger("Encoder")

    def forward(self, x):
        # data must be a clip of "origin" field
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # shape of x5 is [batch_size, 1024, *, *]

        return {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
        }


class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        self.output_dim = output_channels
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, self.output_dim)

    def forward(self, x):
        # data must be the output of encoder
        x1, x2, x3, x4, x5 = x.values()
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Decomposer(nn.Module):
    def __init__(self, input_channels, output_channels, feature_channels):
        super(Decomposer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.feature_channels = feature_channels

        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(output_channels)
        self.reconstruct_loss = ReconstructionLoss()
    from typing import Dict

    def forward(self, data: Dict[str, torch.Tensor]):
        """decompose the T1 or T2 weighted image

        Args:
            data (Dict[str, Tensor]): should contain the following fields:
                "image": the input image
                "target": the target image
                "type": "T1" or "T2"

        Returns:
            dict: a dictionary containing the following fields:
                "loss": the loss of the model
        """
        x = data["image"]
        target = data["target"]
        category = data["type"]
        features = self.encoder(x)
        proton, mapping = self.decoder(features).split(1, dim=1)
        if category == "T1":
            reconstructed = proton * (1 - torch.exp(-mapping))
        elif category == "T2":
            reconstructed = proton * torch.exp(-mapping)
        else:
            raise ValueError(f"Unknown category: {category}")
        # print(reconstructed.shape, target.shape, self.reconstruct_loss)
        loss = self.reconstruct_loss(reconstructed, target)
        return {"loss": loss,
                "map": mapping,
                "proton": proton}


class Enhancer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Enhancer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.enhancer = nn.Sequential(
            Encoder(input_channels),
            Decoder(output_channels),
        )

        self.loss = ReconstructionLoss()

    def forward(self, data):
        """an enhancement net to enhance the T1 map

        Args:
            data (_type_): _description_
        """
        map = data["map"]
        proton = data["proton"]
        target = data["target"]

        enhanced_map = self.enhancer(map)
        reconstructed = proton * (1 - torch.exp(-enhanced_map))
        loss = self.loss(target, reconstructed)
        return {"loss": loss,
                "map": enhanced_map,
                "generated": reconstructed}


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.decomposer = Decomposer(
            input_channels, output_channels * 2, input_channels * 32)
        self.enhancer = Enhancer(input_channels, output_channels)
        self.NH_loss = ReconstructionLoss()
        self.T1CE_loss = ReconstructionLoss()

    def forward(self, data, local, mode="train"):
        t1_weighted = data["T1"]
        t2_weighted = data["T2"]
        t1_enhanced = data["T1CE"]
        loss = 0

        if mode == "train":
            t2_loss, T2, N2 = self.decomposer({
                "image": t2_weighted,
                "target": t2_weighted,
                "type": "T2"
            }).values()
            loss += t2_loss

            t1ce_loss, T1CE_descomposed, N1CE = self.decomposer({
                "image": t1_enhanced,
                "target": t1_enhanced,
                "type": "T1"
            }).values()
            loss += t1ce_loss
            loss += self.NH_loss(N2, N1CE)

            del T2

        t1_loss, T1, N1 = self.decomposer({
            "image": t1_weighted,
            "target": t1_weighted,
            "type": "T1"
        }).values()
        loss += t1_loss
        if mode == "train":
            loss += self.NH_loss(N1CE, N1)
            loss += self.NH_loss(N2, N1)
            del N2

        enhanced_loss, T1CE_enhanced, enhanced = self.enhancer({
            "map": T1,
            "proton": N1,
            "target": t1_enhanced
        }).values()
        loss += enhanced_loss

        if mode == "train":
            symbosis_loss = self.T1CE_loss(
                T1CE_enhanced, T1CE_descomposed) * 0.1
            loss += symbosis_loss
            if local is not None:
                writer = local.writer
                global_step = local.global_step
                writer.add_scalar("loss/t1", t1_loss, global_step)
                writer.add_scalar("loss/t2", t2_loss, global_step)
                writer.add_scalar("loss/t1ce", t1ce_loss, global_step)
                writer.add_scalar("loss/enhanced", enhanced_loss, global_step)
                writer.add_scalar("loss/symbosis", symbosis_loss, global_step)

        return {
            "loss": loss,
            "enhanced": enhanced
        }
