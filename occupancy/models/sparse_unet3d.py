__all__ = ["sparse_unet_encoder4x", "sparse_unet_encoder8x"]

import torch.nn as nn
from spconv.pytorch import (
    SparseReLU,
    SparseBatchNorm,
    SparseConv3d,
    SubMConv3d,
    SparseInverseConv3d,
    SparseSequential,
    SparseConvTensor,
    SparseModule,
)


def post_act_block(
    in_channels,
    out_channels,
    kernel_size,
    indice_key=None,
    stride=1,
    padding=0,
    conv_type="subm",
):
    if conv_type == "subm":
        conv = SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key
        )
    elif conv_type == "spconv":
        conv = SparseConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key,
        )
    elif conv_type == "inverseconv":
        conv = SparseInverseConv3d(
            in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False
        )
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        SparseBatchNorm(out_channels),
        SparseReLU(inplace=True),
    )

    return m


class SparseBasicBlock(SparseModule):
    def __init__(self, inplanes, planes, stride=1, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        self.net = SparseSequential(
            SubMConv3d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            SparseBatchNorm(planes),
            SparseReLU(inplace=True),
            SubMConv3d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            SparseBatchNorm(planes),
        )

        self.relu = SparseReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SparseVoxelEncoder4x(nn.Module):
    def __init__(
        self,
        input_channel,
        norm_cfg,
        base_channel,
        out_channel,
        sparse_shape_xyz,
        **kwargs,
    ):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = SparseSequential(
            SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            SparseReLU(inplace=True),
        )

        self.conv1 = SparseSequential(
            SparseBasicBlock(
                base_channel, base_channel, norm_cfg=norm_cfg, indice_key="res1"
            ),
            SparseBasicBlock(
                base_channel, base_channel, norm_cfg=norm_cfg, indice_key="res1"
            ),
        )

        self.conv2 = SparseSequential(
            block(
                base_channel,
                base_channel * 2,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            SparseBasicBlock(
                base_channel * 2, base_channel * 2, norm_cfg=norm_cfg, indice_key="res2"
            ),
            SparseBasicBlock(
                base_channel * 2, base_channel * 2, norm_cfg=norm_cfg, indice_key="res2"
            ),
        )

        self.conv3 = SparseSequential(
            block(
                base_channel * 2,
                base_channel * 4,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            SparseBasicBlock(
                base_channel * 4, base_channel * 4, norm_cfg=norm_cfg, indice_key="res3"
            ),
            SparseBasicBlock(
                base_channel * 4, base_channel * 4, norm_cfg=norm_cfg, indice_key="res3"
            ),
        )

        self.conv_out = SparseSequential(
            SubMConv3d(base_channel * 4, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            SparseReLU(inplace=True),
        )

    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = SparseConvTensor(
            voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)

        x = self.conv_out(x_conv3)

        return {
            "x": x.dense().permute(0, 1, 4, 3, 2),  # B, C, W, H, D
            "pts_feats": [x],
        }


class SparseVoxelEncoder8x(nn.Module):
    def __init__(
        self,
        input_channel,
        norm_cfg,
        base_channel,
        out_channel,
        sparse_shape_xyz,
        **kwargs,
    ):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = SparseSequential(
            SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            SparseReLU(inplace=True),
        )

        self.conv1 = SparseSequential(
            block(
                base_channel,
                base_channel * 2,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key="spconv1",
                conv_type="spconv",
            ),
            SparseBasicBlock(
                base_channel * 2, base_channel * 2, norm_cfg=norm_cfg, indice_key="res1"
            ),
            SparseBasicBlock(
                base_channel * 2, base_channel * 2, norm_cfg=norm_cfg, indice_key="res1"
            ),
        )

        self.conv2 = SparseSequential(
            block(
                base_channel * 2,
                base_channel * 4,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            SparseBasicBlock(
                base_channel * 4, base_channel * 4, norm_cfg=norm_cfg, indice_key="res2"
            ),
            SparseBasicBlock(
                base_channel * 4, base_channel * 4, norm_cfg=norm_cfg, indice_key="res2"
            ),
        )

        self.conv3 = SparseSequential(
            block(
                base_channel * 4,
                base_channel * 8,
                3,
                norm_cfg=norm_cfg,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            SparseBasicBlock(
                base_channel * 8, base_channel * 8, norm_cfg=norm_cfg, indice_key="res3"
            ),
            SparseBasicBlock(
                base_channel * 8, base_channel * 8, norm_cfg=norm_cfg, indice_key="res3"
            ),
        )

        self.conv_out = SparseSequential(
            SubMConv3d(base_channel * 8, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            SparseReLU(inplace=True),
        )

    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = SparseConvTensor(
            voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)

        x = self.conv_out(x_conv3)

        return {
            "x": x.dense().permute(0, 1, 4, 3, 2),  # B, C, W, H, D
            "pts_feats": [x],
        }


def sparse_unet_encoder4x(**kwargs) -> nn.Module:
    return SparseVoxelEncoder4x(**kwargs)


def sparse_unet_encoder8x(**kwargs) -> nn.Module:
    return SparseVoxelEncoder8x(**kwargs)
