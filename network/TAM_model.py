import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        padding = (0, kernel_size // 2, kernel_size // 2)
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class TemporalAdaptiveModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TemporalAdaptiveModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TAM_EventBranch(nn.Module):
    def __init__(self, pretrained=False):
        super(TAM_EventBranch, self).__init__()
        self.base_model = r3d_18(pretrained=pretrained)
        # Modify stem to accept 2 channels (Polarity)
        self.base_model.stem[0] = nn.Conv3d(2, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                                            bias=False)

        self.sam2 = SpatialAttentionModule()
        self.se2 = SEModule(128)
        self.tam2 = TemporalAdaptiveModule(128)

        self.sam4 = SpatialAttentionModule()
        self.se4 = SEModule(512)
        self.tam4 = TemporalAdaptiveModule(512)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.feature_dim = 256

    def forward(self, x):
        x = self.base_model.stem(x)
        x = self.base_model.layer1(x)
        out_l2 = self.base_model.layer2(x)
        out_l2_attended = self.tam2(self.se2(self.sam2(out_l2)))
        out_l3 = self.base_model.layer3(out_l2)
        out_l4 = self.base_model.layer4(out_l3)
        out_l4_attended = self.tam4(self.se4(self.sam4(out_l4)))

        out_l2_final = torch.mean(out_l2_attended, dim=2)
        out_l4_final = torch.mean(out_l4_attended, dim=2)

        # Align spatial dimensions of layer4 to layer2
        out_l4_upsampled = F.interpolate(
            out_l4_final, size=out_l2_final.shape[-2:], mode='bilinear', align_corners=False
        )

        fused_features_2d = torch.cat([out_l2_final, out_l4_upsampled], dim=1)
        fused_features_2d = self.fusion_conv(fused_features_2d)

        final_vec = F.adaptive_avg_pool2d(fused_features_2d, (1, 1))
        final_vec = final_vec.view(final_vec.size(0), -1)

        return final_vec


class MS_STANet_Pose(nn.Module):
    """
    Dual-stream network fusing Event (MS-STANet) and Pose (Bi-LSTM) modalities.
    """
    def __init__(self, num_classes=10, pose_input_dim=52, pose_hidden_dim=128, pose_rnn_layers=2, dropout=0.5):
        super(MS_STANet_Pose, self).__init__()

        # --- 1. Event Branch (MS-STANet backbone) ---
        self.event_branch = TAM_EventBranch(pretrained=True)
        event_feature_dim = self.event_branch.feature_dim

        # --- 2. Pose Branch (Bi-Directional LSTM) ---
        self.pose_branch = nn.LSTM(
            input_size=pose_input_dim,
            hidden_size=pose_hidden_dim,
            num_layers=pose_rnn_layers,
            batch_first=True,
            dropout=dropout if pose_rnn_layers > 1 else 0,
            bidirectional=True
        )

        # Temporal attention for pose aggregation
        self.pose_attn = nn.Sequential(
            nn.Linear(pose_hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.pose_dropout = nn.Dropout(dropout)

        # --- 3. Fusion & Classification ---
        pose_output_dim = pose_hidden_dim * 2  # Bidirectional

        # Asymmetric Gated Fusion
        self.gate_mlp = nn.Sequential(
            nn.Linear(event_feature_dim + pose_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, event_feature_dim + pose_output_dim),
            nn.Sigmoid()
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(event_feature_dim + pose_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        # Auxiliary heads for supervision
        self.event_aux_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(event_feature_dim, num_classes)
        )
        self.pose_aux_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pose_output_dim, num_classes)
        )

    def forward(self, event_input, pose_input, return_features: bool = False):
        # --- Event Branch Forward ---
        event_features = self.event_branch(event_input)

        # --- Pose Branch Forward ---
        self.pose_branch.flatten_parameters()
        pose_seq, _ = self.pose_branch(pose_input)
        
        # Attention-based aggregation
        attn_logits = self.pose_attn(pose_seq)
        attn = torch.softmax(attn_logits, dim=1)
        pose_features = torch.sum(attn * pose_seq, dim=1)
        pose_features = self.pose_dropout(pose_features)

        # --- Cross-Modal Gating Fusion ---
        combined_features = torch.cat((event_features, pose_features), dim=1)
        gates = self.gate_mlp(combined_features)
        gate_event, gate_pose = torch.split(gates, [event_features.size(1), pose_features.size(1)], dim=1)
        
        # Apply gating (Asymmetric: Event * Gate, Pose * (1-Gate))
        event_gated = event_features * gate_event
        pose_gated = pose_features * (1.0 - gate_pose)
        combined_features = torch.cat((event_gated, pose_gated), dim=1)

        # --- Classification ---
        output = self.fusion_classifier(combined_features)

        if return_features:
            event_logits = self.event_aux_head(event_features)
            pose_logits = self.pose_aux_head(pose_features)
            return output, event_logits, pose_logits
        else:
            return output


def TAM(num_classes=10, pretrained=False):
    """
    Legacy factory function for compatibility.
    """
    print("Notice: Calling TAM() now returns the MS_STANet_Pose dual-stream model.")
    return MS_STANet_Pose(num_classes=num_classes)