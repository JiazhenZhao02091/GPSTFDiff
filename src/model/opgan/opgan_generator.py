import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from .cnnblock import ResidualBlock


from typing import Optional


class OPGANGenerator(nn.Module):
    def __init__(self, img_channel_num: int = 6):
        super().__init__()

        self.TCCB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=9, padding=4), nn.ReLU()
        )

        self.TCCB_RBs = nn.ModuleList()
        for _ in range(8):
            self.TCCB_RBs.append(ResidualBlock(64))
        self.TCCB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )


        self.SDEB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=9, padding=4), nn.ReLU()
        )

        self.SDEB_RBs = nn.ModuleList()
        for _ in range(4):
            self.SDEB_RBs.append(ResidualBlock(64))
        self.SDEB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )


        self.BILB_RBs_head = nn.Sequential(
            nn.Conv2d(img_channel_num, 64, kernel_size=9, padding=4), nn.ReLU()
        )

        self.BILB_RBs = nn.ModuleList()
        for _ in range(2):
            self.BILB_RBs.append(ResidualBlock(64))
        self.BILB_RBs_tail = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(64, img_channel_num, 9, 1, padding='same'), nn.Tanh()
        )

        self.temporal_change_tail = nn.Sequential(
            nn.Conv2d(64, img_channel_num, 9, 1, padding='same'), nn.Tanh()
        )

    def forward(
        self,
        coarse_img_01: torch.tensor,
        coarse_img_02: torch.tensor,
        fine_img_01: torch.tensor,
    ) -> torch.tensor:
        coarse_temporal_change_img = coarse_img_02 - coarse_img_01
        coarse_temporal_change_feats = self.TCCB_RBs_head(coarse_temporal_change_img)
        coarse_temporal_change_feats_residuals = coarse_temporal_change_feats
        for i in range(8):
            coarse_temporal_change_feats_residuals = self.TCCB_RBs[i](
                coarse_temporal_change_feats_residuals
            )
        coarse_temporal_change_feats_residuals = self.TCCB_RBs_tail(
            coarse_temporal_change_feats_residuals
        )
        coarse_temporal_change_feats = (
            coarse_temporal_change_feats + coarse_temporal_change_feats_residuals
        )


        sensor_difference_img = fine_img_01 - coarse_img_01
        sensor_difference_feats = self.SDEB_RBs_head(sensor_difference_img)
        sensor_difference_feats_residuals = sensor_difference_feats
        for i in range(4):
            sensor_difference_feats_residuals = self.SDEB_RBs[i](
                sensor_difference_feats_residuals
            )
        sensor_difference_feats_residuals = self.SDEB_RBs_tail(
            sensor_difference_feats_residuals
        )
        sensor_difference_feats = (
            sensor_difference_feats + sensor_difference_feats_residuals
        )


        base_information_img = fine_img_01
        base_information_feats = self.BILB_RBs_head(base_information_img)
        base_information_feats_residuals = base_information_feats
        for i in range(2):
            base_information_feats_residuals = self.BILB_RBs[i](
                base_information_feats_residuals
            )
        base_information_feats_residuals = self.BILB_RBs_tail(
            base_information_feats_residuals
        )
        base_information_feats = (
            base_information_feats + base_information_feats_residuals
        )


        fine_temporal_change_feats = (
            coarse_temporal_change_feats + sensor_difference_feats
        )
        fine_temporal_change_img = self.temporal_change_tail(fine_temporal_change_feats)


        fine_feats = (
            coarse_temporal_change_feats
            + sensor_difference_feats
            + base_information_feats
        )
        fine_img = self.tail(fine_feats)

        return fine_img, fine_temporal_change_img
