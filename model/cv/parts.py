import logging
import torch
import torch.nn as nn
from model.unet.unet_parts import *
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerDecoderLayer


class EncoderAttentionLayer(nn.Module):
    def __init__(self, input_channels):
        super(EncoderAttentionLayer, self).__init__()
        self.input_dim = input_channels
        assert self.input_dim % 8 == 0, "input dimension must be divisible by 8"
        self.tokenizer = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(start_dim=2, end_dim=-1),
            nn.Linear(16 * 16, 32),
        )
        self.attention_layer = TransformerEncoderLayer(
            d_model=32, nhead=8)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        token = self.tokenizer(x)
        token = self.attention_layer(token)
        return self.pooling(token).unsqueeze(-1)


class DecoderAttentionLayer(nn.Module):
    def __init__(self, input_channels):
        super(DecoderAttentionLayer, self).__init__()
        self.input_dim = input_channels

    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.input_dim = input_channels

        self.inc = DoubleConv(self.input_dim, self.input_dim * 16)
        self.down1 = Down(self.input_dim * 16, self.input_dim * 32)
        self.attention1 = EncoderAttentionLayer(self.input_dim * 32)
        self.down2 = Down(self.input_dim * 32, self.input_dim * 64)
        self.attention2 = EncoderAttentionLayer(self.input_dim * 64)
        self.down3 = Down(self.input_dim * 64, self.input_dim * 128)
        self.attention3 = EncoderAttentionLayer(self.input_dim * 128)
        # self.down4 = Down(self.input_dim * 128, self.input_dim * 256)
        # self.attention4 = EncoderAttentionLayer(self.input_dim * 256)
        self.logger = logging.getLogger("Encoder")

    def forward(self, x):
        # data must be a clip of "origin" field
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = x2 * self.attention1(x2)
        x3 = self.down2(x2)
        x3 = x3 * self.attention2(x3)
        x4 = self.down3(x3)
        x4 = x4 * self.attention3(x4)
        # x5 = self.down4(x4)  # shape of x5 is [batch_size, 1024, *, *]
        # x5 = x5 * self.attention4(x5)

        return {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            # "x5": x5,
        }


class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        self.output_dim = output_channels
        # self.up1 = Up(self.output_dim * 256, self.output_dim * 128)
        self.up2 = Up(self.output_dim * 128, self.output_dim * 64)
        self.up3 = Up(self.output_dim * 64, self.output_dim * 32)
        self.up4 = Up(self.output_dim * 32, self.output_dim * 16)
        self.outc = OutConv(self.output_dim * 16, self.output_dim)

    def forward(self, x):
        # data must be the output of encoder
        x1, x2, x3, x4 = x.values()
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ElementLoss(nn.Module):
    def __init__(self):
        super(ElementLoss, self).__init__()
        # 6 main elements in the head
        self.elements = nn.parameter.Parameter(torch.randn(6))

    def forward(self, x):
        loss = 1
        for e in self.elements:
            loss *= (x - e)
        return torch.linalg.norm(loss)


class Decomposer(nn.Module):
    def __init__(self, input_channels, output_channels, feature_channels):
        super(Decomposer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.feature_channels = feature_channels

        self.encoder = Encoder(input_channels)
        self.map_decoder = Decoder(output_channels)
        self.proton_decoder = Decoder(output_channels)

        self.reconstruct_loss = nn.L1Loss()
        self.formalized_loss = nn.ModuleList(
            ElementLoss() for _ in range(2)
        )
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
        mapping = self.map_decoder(features)
        proton = self.proton_decoder(features)
        if category == "T1":
            reconstructed = proton * (1 - torch.exp(-mapping))
        elif category == "T2":
            reconstructed = proton * torch.exp(-mapping)
        else:
            raise ValueError(f"Unknown category: {category}")
        # print(reconstructed.shape, target.shape, self.reconstruct_loss)
        loss = self.reconstruct_loss(reconstructed, target)
        # loss += 0.001 * (
        #     self.formalized_loss[0](proton) +
        #     self.formalized_loss[1](mapping)
        # ) / 2
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

        self.loss = nn.L1Loss()

        self.formalized_loss = ElementLoss()

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
        # loss += self.formalized_loss(enhanced_map) * 0.001
        return {"loss": loss,
                "map": enhanced_map,
                "generated": reconstructed}


class GramLoss(nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()
        self.mse_criterion = nn.MSELoss()

    def forward(self, features, targets, weights=None):
        if weights is None:
            weights = [1/len(features)] * len(features)

        gram_loss = 0
        for f, t, w in zip(features, targets, weights):
            gram_loss += self.mse_criterion(self.gram(f), self.gram(t)) * w
        return gram_loss

    def gram(self, x):
        b, c, h, w = x.size()
        g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1, 2))
        return g.div(h*w)


class Classifier(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Classifier, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes

        self.encoder = Encoder(input_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_channels * 128, n_classes),
            nn.Softmax(dim=1)
        )

        self.cross_entropy = nn.CrossEntropyLoss()
        self.gram_loss = GramLoss()

    def forward(self, data):
        """classify the image into one of the classes

        Args:
            data (Dict[str, Tensor]): should contain the following fields:
                "fake": the fake image
                "real": the real image



        Returns:
            dict: the loss of the classifier
        """
        fake = data["fake"]
        real = data["real"]

        f_features = list(self.encoder(fake).values())
        r_features = list(self.encoder(real).values())

        fake_label = self.classifier(f_features[-1])
        real_label = self.classifier(r_features[-1])

        # the cross entropy loss
        loss = (self.cross_entropy(fake_label,
                                   torch.zeros(fake_label.shape[0],
                                               dtype=torch.long,
                                               device=fake_label.device))
                + self.cross_entropy(real_label,
                                     torch.ones(real_label.shape[0],
                                                dtype=torch.long,
                                                device=real_label.device))) / 2

        # the gram loss
        loss += self.gram_loss(f_features, r_features)

        return {"loss": loss,
                "fake_label": fake_label,
                "real_label": real_label}


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.classifier = Classifier(input_channels, 2)

    def forward(self, data):
        fake = data["fake"].detach()
        real = data["real"].detach()
        return self.classifier({"fake": fake, "real": real})


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, discriminator):
        super(Generator, self).__init__()
        self.decomposer = Decomposer(
            input_channels, output_channels, input_channels * 32)
        self.enhancer = Enhancer(input_channels, output_channels)
        self.NH_loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.discriminator = discriminator
        self.T1CE_loss = nn.L1Loss()

    def forward(self, data, mode="train"):
        t1_weighted = data["T1"]
        t2_weighted = data["T2"]
        t1_enhanced = data["T1CE"]
        loss = 0
        _loss, T1, N1 = self.decomposer({
            "image": t1_weighted,
            "target": t1_weighted,
            "type": "T1"
        }).values()
        loss += _loss

        if mode == "train":
            _loss, T2, N2 = self.decomposer({
                "image": t2_weighted,
                "target": t2_weighted,
                "type": "T2"
            }).values()
            loss += _loss
            loss += self.NH_loss(N1, N2)

            _loss, T1CE_descomposed, N1CE = self.decomposer({
                "image": t1_enhanced,
                "target": t1_enhanced,
                "type": "T1"
            }).values()
            loss += _loss
            loss += self.NH_loss(N1, N1CE)

        _loss, T1CE_enhanced, enhanced = self.enhancer({
            "map": T1,
            "proton": N1,
            "target": t1_enhanced
        }).values()
        loss += _loss

        if mode == "train":
            loss += self.T1CE_loss(T1CE_enhanced.detach(), T1CE_descomposed)

        _loss, fake_label, real_label = self.discriminator({
            "fake": enhanced,
            "real": t1_enhanced
        }).values()
        loss += self.cross_entropy(fake_label, torch.ones(
            enhanced.shape[0], dtype=torch.long, device=enhanced.device))

        return {
            "loss": loss,
            "enhanced": enhanced
        }
