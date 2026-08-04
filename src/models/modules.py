"""
Problem Statement (proposal §1) 의 구성요소를 그대로 모듈화합니다.

  z(t) in R^{K x R}                       : LatentState
  r_theta(z)                                : ReactionNetwork  (축 간 결합)
  L_phi (learnable graph Laplacian)         : GraphDiffusionOperator (공간 전파)
  x_m ~ P_psi_m(x_m | Pi_m z)               : ModalityEncoder / ModalityDecoder
"""
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """
    관측 x_m -> latent axis 부분공간으로의 posterior 초기화용 인코더.
    z0의 amortized inference q(z0 | x_m^{baseline}) 에 사용.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityDecoder(nn.Module):
    """
    P_psi_m(x_m | Pi_m z) 의 mean/logvar를 내는 emission head.
    Pi_m 은 축(axis) 선택 + (필요시) ROI->scalar 요약을 포함한 학습 가능한
    projection으로 구현합니다 (완전 고정하지 않고 약간의 자유도를 줌).
    """
    def __init__(self, latent_flat_dim: int, hidden_dim: int, out_dim: int, roi_level: bool):
        super().__init__()
        self.roi_level = roi_level
        self.proj = nn.Sequential(
            nn.Linear(latent_flat_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, z_flat):
        h = self.proj(z_flat)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-6, 6)
        return mean, logvar


class RoiModalityDecoder(nn.Module):
    """
    MRI/PET처럼 ROI 단위로 값을 내는 모달리티용 디코더.
    입력 (B, R, D_in) -> 각 ROI를 독립적으로(가중치는 공유) 스칼라로 매핑.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, z_roi):
        h = self.net(z_roi)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).clamp(-6, 6).squeeze(-1)
        return mean, logvar


class ReactionNetwork(nn.Module):
    """
    r_theta(z(t)): 병리 축(A/T/I/N/V) 간 비선형 상호작용.
    ROI를 배치 차원처럼 취급하고, 축(K=5)을 시퀀스로 보는 작은 Transformer로
    "어떤 축이 어떤 축에 영향을 주는지"를 attention으로 학습하게 합니다.
    reaction_type="mlp" 이면 훨씬 가벼운 대안(각 ROI별 독립 MLP)을 씁니다.
    """
    def __init__(self, n_axes: int, dim_per_axis: int, hidden_dim: int,
                 reaction_type: str = "transformer", n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_axes = n_axes
        self.dim_per_axis = dim_per_axis
        self.reaction_type = reaction_type

        if reaction_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_per_axis, nhead=n_heads, dim_feedforward=hidden_dim,
                batch_first=True, activation="gelu",
            )
            self.axis_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.out_proj = nn.Linear(dim_per_axis, dim_per_axis)
        elif reaction_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(n_axes * dim_per_axis, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_axes * dim_per_axis),
            )
        else:
            raise ValueError(f"unknown reaction_type: {reaction_type}")

    def forward(self, z):
        """
        z: (B, K, R, D)  ->  reaction: (B, K, R, D)
        """
        B, K, R, D = z.shape
        if self.reaction_type == "transformer":
            # (B, K, R, D) -> (B*R, K, D): 각 ROI를 독립적인 "샘플"로 보고
            # 축(K)을 시퀀스로 하는 self-attention 적용
            z_r = z.permute(0, 2, 1, 3).reshape(B * R, K, D)
            h = self.axis_transformer(z_r)
            out = self.out_proj(h)
            out = out.reshape(B, R, K, D).permute(0, 2, 1, 3)
        else:
            z_flat = z.permute(0, 2, 1, 3).reshape(B, R, K * D)
            out = self.mlp(z_flat).reshape(B, R, K, D).permute(0, 2, 1, 3)
        return out - z  # residual reaction (변화율로 해석)


class GraphDiffusionOperator(nn.Module):
    """
    -L_phi z(t) 항. 기본 그래프(고정 adjacency 또는 identity)에 학습 가능한
    edge reweighting을 곱해 patient-population-shared L_phi를 만듭니다.
    (환자별 완전 개인화 L_phi^(i) 로 확장하려면 forward에 patient embedding을
    받아 edge weight를 modulate하는 조건부 형태로 바꾸면 됩니다 — 아래
    `conditioning_dim`가 그 확장 지점입니다.)
    """
    def __init__(self, n_rois: int, base_adjacency: torch.Tensor = None,
                 learnable: bool = True, conditioning_dim: int = 0):
        super().__init__()
        self.n_rois = n_rois
        if base_adjacency is None:
            base_adjacency = torch.eye(n_rois) * 0.0  # 고립 노드 (diffusion 없음)이 기본값
        self.register_buffer("base_adj", base_adjacency)

        self.learnable = learnable
        if learnable:
            # symmetric edge reweighting logits (population-shared)
            self.edge_logits = nn.Parameter(torch.zeros(n_rois, n_rois))

        self.conditioning_dim = conditioning_dim
        if conditioning_dim > 0:
            # 환자 임베딩 -> 전역 diffusion 강도 스칼라 (개인화의 최소 버전)
            self.strength_head = nn.Sequential(
                nn.Linear(conditioning_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus()
            )

    def build_laplacian(self, patient_embed: torch.Tensor = None):
        adj = self.base_adj
        if self.learnable:
            w = torch.sigmoid(self.edge_logits)
            w = (w + w.t()) / 2  # symmetric
            adj = adj + w
        deg = torch.diag(adj.sum(dim=-1))
        L = deg - adj  # (R, R)

        strength = 1.0
        if patient_embed is not None and self.conditioning_dim > 0:
            strength = self.strength_head(patient_embed)  # (B, 1)
        return L, strength

    def forward(self, z, patient_embed: torch.Tensor = None):
        """
        z: (B, K, R, D) -> diffusion term: (B, K, R, D)
        """
        L, strength = self.build_laplacian(patient_embed)
        # -L z  : R 차원에 대해 행렬곱
        diffusion = -torch.einsum("rs,bksd->bkrd", L, z)
        if isinstance(strength, torch.Tensor):
            diffusion = diffusion * strength.view(-1, 1, 1, 1)
        return diffusion
