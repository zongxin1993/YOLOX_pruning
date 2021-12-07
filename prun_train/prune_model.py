from yolox.models import YOLOXHead
import torch
import torch.nn as nn
from yolox.models.network_blocks import BaseConv, Focus


class Bottleneck_prune(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        cv1in,
        cv1out,
        cv2out,
        shortcut=True,
        act="silu",
    ):
        super().__init__()
        self.conv1 = BaseConv(cv1in, cv1out, 1, stride=1, act=act)
        self.conv2 = BaseConv(cv1out, cv2out, 3, stride=1, act=act)
        self.use_add = shortcut and cv1in == cv2out

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer_prune(nn.Module):
    def __init__(self,
                 cv1in,
                 cv1out,
                 cv2out,
                 cv3out,
                 bottle_args,
                 n=1,
                 depthwise=False,
                 shortcut=True,
                 act="silu",
                 ):
        super().__init__()
        cv3in = bottle_args[-1][-1]
        self.conv1 = BaseConv(cv1in, cv1out, 1, stride=1, act=act)
        self.conv2 = BaseConv(cv1in, cv2out, 1, stride=1, act=act)
        self.conv3 = BaseConv(cv3in+cv2out, cv3out, 1, stride=1, act=act)
        module_list = [
            Bottleneck_prune(*bottle_args[i], shortcut, act=act)
            for i in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


class SPPBottleneck_prune(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, cv1in, cv1out, cv2out, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()

        self.cv1in = cv1in
        self.cv1out = cv1out

        self.conv1 = BaseConv(cv1in, cv1out, 1, 1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2)
                               for x in kernel_sizes])
        self.conv2 = BaseConv(cv1out * (len(kernel_sizes) + 1), cv2out, 1, 1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPDarknet_prune(nn.Module):
    def __init__(self,
                 dep_mul,
                 wid_mul,
                 mask_bn_channel_dict,
                 out_features=("dark3", "dark4", "dark5"),
                 depthwise=False,
                 act="silu",
                 ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        base_depth = max(round(dep_mul * 3), 1)

        self.convIn_from_ = {}
        name_tmp = ""
        # stem
        stem_bn_out = mask_bn_channel_dict["backbone.backbone.stem.conv.bn"]
        self.stem = Focus(3, stem_bn_out, ksize=3, act=act)
        self.convIn_from_["backbone.backbone.stem.conv.bn"] = 3*4

        # dark2
        bottle_args = []
        self.convIn_from_["backbone.backbone.dark2.0.bn"] = "backbone.backbone.stem.conv.bn"
        named_m_base = "backbone.backbone.dark2.1"
        conv0_out = mask_bn_channel_dict["backbone.backbone.dark2.0.bn"]
        chin = [conv0_out]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(base_depth):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            name_tmp = named_m_bottle_cv2_bn
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
        self.convIn_from_[named_cv1_bn] = "backbone.backbone.dark2.0.bn"
        self.convIn_from_[named_cv2_bn] = "backbone.backbone.dark2.0.bn"
        self.convIn_from_[named_cv3_bn] = ["backbone.backbone.dark2.1.conv2.bn", name_tmp]
        self.dark2 = nn.Sequential(
            BaseConv(stem_bn_out, conv0_out, 3, 2, act=act),
            CSPLayer_prune(
                conv0_out,
                mask_bn_channel_dict[named_cv1_bn],
                mask_bn_channel_dict[named_cv2_bn],
                mask_bn_channel_dict[named_cv3_bn],
                bottle_args,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        # dark3
        bottle_args = []
        named_m_base = "backbone.backbone.dark3.1"
        conv0_out = mask_bn_channel_dict["backbone.backbone.dark3.0.bn"]
        chin = [conv0_out]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        self.convIn_from_["backbone.backbone.dark3.0.bn"] = "backbone.backbone.dark2.1.conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(base_depth*3):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            name_tmp = named_m_bottle_cv2_bn
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
        self.convIn_from_[named_cv1_bn] = "backbone.backbone.dark3.0.bn"
        self.convIn_from_[named_cv2_bn] = "backbone.backbone.dark3.0.bn"
        self.convIn_from_[named_cv3_bn] = ["backbone.backbone.dark3.1.conv2.bn", name_tmp]
        self.dark3 = nn.Sequential(
            BaseConv(
                mask_bn_channel_dict["backbone.backbone.dark2.1.conv3.bn"], conv0_out, 3, 2, act=act),
            CSPLayer_prune(
                conv0_out,
                mask_bn_channel_dict[named_cv1_bn],
                mask_bn_channel_dict[named_cv2_bn],
                mask_bn_channel_dict[named_cv3_bn],
                bottle_args,
                n=base_depth*3,
                depthwise=depthwise,
                act=act,
            ),
        )
        # dark4
        bottle_args = []
        named_m_base = "backbone.backbone.dark4.1"
        conv0_out = mask_bn_channel_dict["backbone.backbone.dark4.0.bn"]
        chin = [conv0_out]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        self.convIn_from_["backbone.backbone.dark4.0.bn"] = "backbone.backbone.dark3.1.conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(base_depth*3):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
            name_tmp = named_m_bottle_cv2_bn
        self.convIn_from_[named_cv1_bn] = "backbone.backbone.dark4.0.bn"
        self.convIn_from_[named_cv2_bn] = "backbone.backbone.dark4.0.bn"
        self.convIn_from_[named_cv3_bn] = ["backbone.backbone.dark4.1.conv2.bn", name_tmp]
        self.dark4 = nn.Sequential(
            BaseConv(
                mask_bn_channel_dict["backbone.backbone.dark3.1.conv3.bn"], conv0_out, 3, 2, act=act),
            CSPLayer_prune(
                conv0_out,
                mask_bn_channel_dict[named_cv1_bn],
                mask_bn_channel_dict[named_cv2_bn],
                mask_bn_channel_dict[named_cv3_bn],
                bottle_args,
                n=base_depth*3,
                depthwise=depthwise,
                act=act,
            ),
        )
        # dark5
        bottle_args = []
        named_m_base = "backbone.backbone.dark5.2"
        conv0_out = mask_bn_channel_dict["backbone.backbone.dark5.0.bn"]
        chin = [conv0_out]
        self.convIn_from_["backbone.backbone.dark5.0.bn"] = "backbone.backbone.dark4.1.conv3.bn"
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(base_depth):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
            name_tmp = named_m_bottle_cv2_bn

        ssp_named_cv1_bn = "backbone.backbone.dark5.1.conv1.bn"
        ssp_named_cv2_bn = "backbone.backbone.dark5.1.conv2.bn"
        ssp_cv1out = mask_bn_channel_dict[ssp_named_cv1_bn]
        ssp_cv2out = mask_bn_channel_dict[ssp_named_cv2_bn]
        self.convIn_from_[ssp_named_cv1_bn] = "backbone.backbone.dark5.0.bn"
        self.convIn_from_[ssp_named_cv2_bn] = [ssp_named_cv1_bn]*4

        self.convIn_from_[named_cv1_bn] = ssp_named_cv2_bn
        self.convIn_from_[named_cv2_bn] = ssp_named_cv2_bn
        self.convIn_from_[named_cv3_bn] = ["backbone.backbone.dark5.2.conv2.bn", name_tmp]

        self.dark5 = nn.Sequential(
            BaseConv(
                mask_bn_channel_dict["backbone.backbone.dark4.1.conv3.bn"], conv0_out, 3, 2, act=act),
            SPPBottleneck_prune(conv0_out, ssp_cv1out, ssp_cv2out, activation=act),
            CSPLayer_prune(
                ssp_cv2out,
                mask_bn_channel_dict[named_cv1_bn],
                mask_bn_channel_dict[named_cv2_bn],
                mask_bn_channel_dict[named_cv3_bn],
                bottle_args,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def get_convIn_from(self,):
        return self.convIn_from_


class YOLOPAFPN_prune(nn.Module):
    def __init__(self,
                 mask_bn_channel_dict,
                 in_channels=[256, 512, 1024],
                 in_features=("dark3", "dark4", "dark5"),
                 act="silu",
                 depthwise=False,
                 depth=1,
                 width=1):
        super().__init__()
        self.in_features = in_features
        self.convIn_from_ = {}
        name_tmp = ""
        self.backbone = CSPDarknet_prune(
            depth, width, mask_bn_channel_dict=mask_bn_channel_dict, depthwise=depthwise, act=act)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.convIn_from_["backbone.lateral_conv0.bn"] = "backbone.backbone.dark5.2.conv3.bn"
        lateral_conv0_out = mask_bn_channel_dict["backbone.lateral_conv0.bn"]
        self.lateral_conv0 = BaseConv(
            mask_bn_channel_dict["backbone.backbone.dark5.1.conv2.bn"], lateral_conv0_out, 1, 1, act=act
        )

        bottle_args = []
        named_m_base = "backbone.C3_p4"
        cv1_in = lateral_conv0_out + mask_bn_channel_dict["backbone.backbone.dark4.1.conv3.bn"]
        chin = [cv1_in]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(3*depth):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
            name_tmp = named_m_bottle_cv2_bn
        self.convIn_from_[named_cv1_bn] = ["backbone.lateral_conv0.bn","backbone.backbone.dark4.1.conv3.bn"]
        self.convIn_from_[named_cv2_bn] = ["backbone.lateral_conv0.bn","backbone.backbone.dark4.1.conv3.bn"]
        self.convIn_from_[named_cv3_bn] = ["backbone.C3_p4.conv2.bn", name_tmp]
        self.C3_p4 = CSPLayer_prune(
            cv1_in,
            mask_bn_channel_dict[named_cv1_bn],
            mask_bn_channel_dict[named_cv2_bn],
            mask_bn_channel_dict[named_cv3_bn],
            bottle_args,
            n=3*depth,
            depthwise=depthwise,
            act=act,
        )

        reduce_conv1_out = mask_bn_channel_dict["backbone.reduce_conv1.bn"]
        self.convIn_from_["backbone.reduce_conv1.bn"] = named_cv3_bn
        self.reduce_conv1 = BaseConv(
            bottle_args[-1][-1], reduce_conv1_out, 1, 1, act=act
        )
        bottle_args = []
        named_m_base = "backbone.C3_p3"
        cv1_in = reduce_conv1_out + mask_bn_channel_dict["backbone.backbone.dark3.1.conv3.bn"]
        chin = [cv1_in]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(3*depth):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
            name_tmp = named_m_bottle_cv2_bn
        self.convIn_from_[named_cv1_bn] = ["backbone.reduce_conv1.bn","backbone.backbone.dark3.1.conv3.bn"]
        self.convIn_from_[named_cv2_bn] = ["backbone.reduce_conv1.bn","backbone.backbone.dark3.1.conv3.bn"]
        self.convIn_from_[named_cv3_bn] = ["backbone.C3_p3.conv2.bn", name_tmp]
        self.C3_p3 = CSPLayer_prune(
            cv1_in,
            mask_bn_channel_dict[named_cv1_bn],
            mask_bn_channel_dict[named_cv2_bn],
            mask_bn_channel_dict[named_cv3_bn],
            bottle_args,
            n=3*depth,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        bu_conv2_out = mask_bn_channel_dict["backbone.bu_conv2.bn"]
        self.convIn_from_["backbone.bu_conv2.bn"] = named_cv3_bn
        self.bu_conv2 = BaseConv(
            bottle_args[-1][-1], bu_conv2_out, 3, 2, act=act
        )
        bottle_args = []
        named_m_base = "backbone.C3_n3"
        cv1_in = bu_conv2_out + mask_bn_channel_dict["backbone.reduce_conv1.bn"]
        chin = [cv1_in]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(3*depth):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
            name_tmp = named_m_bottle_cv2_bn
        self.convIn_from_[named_cv1_bn] = ["backbone.bu_conv2.bn","backbone.reduce_conv1.bn"]
        self.convIn_from_[named_cv2_bn] = ["backbone.bu_conv2.bn","backbone.reduce_conv1.bn"]
        self.convIn_from_[named_cv3_bn] = ["backbone.C3_n3.conv2.bn", name_tmp]
        self.C3_n3 = CSPLayer_prune(
            cv1_in,
            mask_bn_channel_dict[named_m_base+".conv1.bn"],
            mask_bn_channel_dict[named_m_base+".conv2.bn"],
            mask_bn_channel_dict[named_m_base+".conv3.bn"],
            bottle_args,
            n=3*depth,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        bu_conv1_out = mask_bn_channel_dict["backbone.bu_conv1.bn"]
        self.convIn_from_["backbone.bu_conv1.bn"] = named_cv3_bn
        self.bu_conv1 = BaseConv(
            bottle_args[-1][-1], bu_conv1_out, 3, 2, act=act
        )
        bottle_args = []
        named_m_base = "backbone.C3_n4"
        cv1_in = bu_conv2_out + mask_bn_channel_dict["backbone.lateral_conv0.bn"]
        chin = [cv1_in]
        named_cv1_bn = named_m_base + ".conv1.bn"
        named_cv2_bn = named_m_base + ".conv2.bn"
        named_cv3_bn = named_m_base + ".conv3.bn"
        c3fromlayer = [named_cv1_bn]
        for p in range(3*depth):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.conv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.conv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = mask_bn_channel_dict[named_m_bottle_cv1_bn]
            bottle_cv2out = mask_bn_channel_dict[named_m_bottle_cv2_bn]
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.convIn_from_[named_m_bottle_cv1_bn] = c3fromlayer[p]
            self.convIn_from_[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
            c3fromlayer.append(named_m_bottle_cv2_bn)
            name_tmp = named_m_bottle_cv2_bn
        self.convIn_from_[named_cv1_bn] = ["backbone.bu_conv1.bn","backbone.lateral_conv0.bn"]
        self.convIn_from_[named_cv2_bn] = ["backbone.bu_conv1.bn","backbone.lateral_conv0.bn"]
        self.convIn_from_[named_cv3_bn] = ["backbone.C3_n4.conv2.bn", name_tmp]
        self.C3_n4 = CSPLayer_prune(
            cv1_in,
            mask_bn_channel_dict[named_cv1_bn],
            mask_bn_channel_dict[named_cv2_bn],
            mask_bn_channel_dict[named_cv3_bn],
            bottle_args,
            n=3*depth,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

    def get_convIn_from(self,):
        return {**self.convIn_from_, **self.backbone.get_convIn_from()}

class YOLOXHead_prune(YOLOXHead):
    def __init__(self,
                 num_classes,
                 mask_bn_channel_dict,
                width=1.0,
                strides=[8, 16, 32],
                in_channels=[256, 512, 1024],
                act="silu",
                depthwise=False,
        ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__(num_classes)

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = BaseConv

        self.convIn_from_ = {}
        self.convIn_from_["head.stems.0.bn"] = "backbone.C3_p3.conv3.bn"
        self.convIn_from_["head.stems.1.bn"] = "backbone.C3_n3.conv3.bn"
        self.convIn_from_["head.stems.2.bn"] = "backbone.C3_n4.conv3.bn"

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(mask_bn_channel_dict[self.convIn_from_["head.stems.{}.bn".format(i)]] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            # self.convIn_from_["head.cls_convs.{}.0.bn".format(i)] = "head.stems.{}.bn".format(i)
            # self.convIn_from_["head.cls_convs.{}.1.bn".format(i)] = "head.cls_convs.{}.0.bn".format(i)
            # self.convIn_from_["head.reg_convs.{}.0.bn".format(i)] = "head.stems.{}.bn".format(i)
            # self.convIn_from_["head.reg_convs.{}.1.bn".format(i)] = "head.reg_convs.{}.0.bn".format(i)

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
    
    def get_convIn_from(self,):
        return self.convIn_from_

class yolox_prun(nn.Module):
    def __init__(self, mask_bn_channel_dict):
        super().__init__()
        backbone = YOLOPAFPN_prune(mask_bn_channel_dict=mask_bn_channel_dict)
        head = YOLOXHead_prune(1,mask_bn_channel_dict)
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

