"""
Proposal §1.5 의

    dz(t) = [-L_phi z(t) + r_theta(z(t))] dt + Sigma^{1/2}(z(t)) dW(t)
    x_m(t) ~ P_psi_m(x_m | Pi_m z(t))

를 그대로 구현한 전체 모델. 비정기 방문(irregular visit times)을
Euler–Maruyama로 적분하며, 각 방문 시점에서 관측된 모달리티만 loss에
반영합니다(마스킹).

단순화 지점(= 다음 단계에서 정교화할 부분, 코드 곳곳에 TODO로 표시):
  - Pi_m(투영)을 엄밀한 선형 사상이 아니라 축(axis) 선택 + 학습 가능한 head로
    근사했습니다. 완전히 고정된 해석 가능한 projection을 원하면
    ModalityDecoder/RoiModalityDecoder를 선형층으로 제한하면 됩니다.
  - identifiability 제약(§1.4)의 anchor/ordering loss는 최소 형태의
    프록시입니다. 실제 ATN 양성 라벨이 확보되면 더 엄밀한 순서 제약으로
    교체해야 합니다.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    ReactionNetwork, GraphDiffusionOperator, ModalityDecoder, RoiModalityDecoder
)

AXIS_INDEX = {"A": 0, "T": 1, "I": 2, "N": 3, "V": 4}


class LatentCoupledReactionDiffusion(nn.Module):
    def __init__(self, model_cfg: dict, n_rois: int, n_axes: int = 5):
        super().__init__()
        self.K = n_axes
        self.R = n_rois
        self.D = model_cfg["latent_dim_per_axis"]
        H = model_cfg["hidden_dim"]
        self.sde_steps = model_cfg["sde_steps_per_interval"]

        # --- baseline encoder input 차원 ---
        # blood(3) + csf(2) + mri(R) + pet(R) + genetics(1) + cognition(2)
        # + 각 모달리티 결측 여부 flag(6)
        self.baseline_dim = 3 + 2 + n_rois + n_rois + 1 + 2 + 6

        z_flat_dim = self.K * self.R * self.D
        self.z0_encoder = nn.Sequential(
            nn.Linear(self.baseline_dim, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
        )
        self.z0_mean_head = nn.Linear(H, z_flat_dim)
        self.z0_logvar_head = nn.Linear(H, z_flat_dim)

        self.patient_embed_net = nn.Sequential(
            nn.Linear(self.baseline_dim, H), nn.GELU(), nn.Linear(H, 32)
        )

        self.reaction = ReactionNetwork(
            n_axes=self.K, dim_per_axis=self.D, hidden_dim=H,
            reaction_type=model_cfg["reaction_type"],
            n_layers=model_cfg["reaction_layers"],
            n_heads=model_cfg["reaction_heads"],
        )

        base_adj = None
        if model_cfg["graph_init"] == "template" and model_cfg.get("adjacency_csv"):
            import pandas as pd
            adj_df = pd.read_csv(model_cfg["adjacency_csv"], header=None)
            base_adj = torch.tensor(adj_df.values, dtype=torch.float32)
            assert base_adj.shape == (n_rois, n_rois), \
                f"adjacency_csv shape {base_adj.shape} != (n_rois, n_rois)=({n_rois},{n_rois})"

        self.graph_diffusion = GraphDiffusionOperator(
            n_rois=n_rois, base_adjacency=base_adj,
            learnable=model_cfg["learnable_graph"], conditioning_dim=32,
        )

        self.log_noise_scale = nn.Parameter(
            torch.log(torch.tensor(float(model_cfg["noise_scale_init"])))
        )

        # --- emission decoders ---
        # 전역(pooled-over-ROI) 모달리티: blood, csf, cognition
        self.blood_decoder = ModalityDecoder(self.K * self.D, H, out_dim=3, roi_level=False)
        self.csf_decoder = ModalityDecoder(self.K * self.D, H, out_dim=2, roi_level=False)
        self.cognition_decoder = ModalityDecoder(self.D, H, out_dim=2, roi_level=False)
        # ROI-level 모달리티: mri(N axis 기반), pet(A,T axis 기반)
        self.mri_decoder = RoiModalityDecoder(self.D, H)
        self.pet_decoder = RoiModalityDecoder(2 * self.D, H)

        # 보조 진단 분류 head (해석 가능성 + 정확도 보조 신호)
        self.diag_head = nn.Sequential(
            nn.Linear(2 * self.D, H), nn.GELU(), nn.Linear(H, 3)
        )

    # ---------- baseline feature 구성 ----------
    def _build_baseline_input(self, batch: Dict) -> torch.Tensor:
        obs, mask = batch["observations"], batch["obs_mask"]

        def masked0(name):
            v0 = obs[name][:, 0]  # (B, d) genetics는 (B, d) 그대로
            m0 = mask[name][:, 0].float()
            if v0.dim() == 2 and m0.dim() == 1:
                m0 = m0.unsqueeze(-1)
            return v0 * m0, mask[name][:, 0].float().mean(dim=-1, keepdim=True) \
                if mask[name].dim() > 2 else mask[name][:, 0].float().unsqueeze(-1)

        blood_v, blood_m = masked0("blood")
        csf_v, csf_m = masked0("csf")
        mri_v, mri_m = masked0("mri")
        pet_v, pet_m = masked0("pet")
        cog_v, cog_m = masked0("cognition")
        gen_v = obs["genetics"]  # (B,1) 1회성, 결측 거의 없다고 가정
        gen_m = mask["genetics"][:, 0].float().unsqueeze(-1) if mask["genetics"].dim() > 1 \
            else mask["genetics"].float().unsqueeze(-1)

        flags = torch.cat([blood_m, csf_m, mri_m, pet_m, gen_m, cog_m], dim=-1)
        baseline = torch.cat([blood_v, csf_v, mri_v, pet_v, gen_v, cog_v, flags], dim=-1)
        return baseline

    # ---------- z0 posterior ----------
    def encode_z0(self, batch: Dict):
        baseline = self._build_baseline_input(batch)
        h = self.z0_encoder(baseline)
        mean = self.z0_mean_head(h).view(-1, self.K, self.R, self.D)
        logvar = self.z0_logvar_head(h).clamp(-6, 6).view(-1, self.K, self.R, self.D)
        patient_embed = self.patient_embed_net(baseline)
        return mean, logvar, patient_embed

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    # ---------- SDE 적분 (Euler–Maruyama, 비정기 방문 지원) ----------
    def integrate(self, z0: torch.Tensor, visit_months: torch.Tensor,
                  patient_embed: torch.Tensor):
        B, T = visit_months.shape
        traj = [z0]
        z = z0
        sigma = torch.exp(self.log_noise_scale)

        for t in range(1, T):
            dt_total = (visit_months[:, t] - visit_months[:, t - 1]).clamp(min=0.0)
            dt_step = (dt_total / self.sde_steps).view(-1, 1, 1, 1)  # (B,1,1,1)
            for _ in range(self.sde_steps):
                diffusion = self.graph_diffusion(z, patient_embed)
                reaction = self.reaction(z)
                drift = diffusion + reaction
                noise = sigma * torch.sqrt(dt_step.clamp(min=1e-8)) * torch.randn_like(z)
                z = z + dt_step * drift + noise
            traj.append(z)
        return torch.stack(traj, dim=1)  # (B, T, K, R, D)

    # ---------- emission ----------
    def decode(self, z_traj: torch.Tensor):
        """
        z_traj: (B, T, K, R, D)
        반환: 모달리티별 (mean, logvar), shape (B,T,out_dim) 또는 (B,T,R)
        """
        pooled = z_traj.mean(dim=3)  # (B, T, K, D) : ROI 평균 (전역 모달리티용)
        B, T, K, D = pooled.shape
        pooled_flat = pooled.reshape(B * T, K * D)

        blood_mean, blood_logvar = self.blood_decoder(pooled_flat)
        csf_mean, csf_logvar = self.csf_decoder(pooled_flat)

        n_axis_pooled = pooled[:, :, AXIS_INDEX["N"], :].reshape(B * T, D)
        cog_mean, cog_logvar = self.cognition_decoder(n_axis_pooled)

        n_axis_roi = z_traj[:, :, AXIS_INDEX["N"], :, :].reshape(B * T, self.R, D)
        mri_mean, mri_logvar = self.mri_decoder(n_axis_roi)

        at_roi = torch.cat(
            [z_traj[:, :, AXIS_INDEX["A"], :, :], z_traj[:, :, AXIS_INDEX["T"], :, :]], dim=-1
        ).reshape(B * T, self.R, 2 * D)
        pet_mean, pet_logvar = self.pet_decoder(at_roi)

        def reshape_global(x, out_dim):
            return x.view(B, T, out_dim)

        def reshape_roi(x):
            return x.view(B, T, self.R)

        out = {
            "blood": (reshape_global(blood_mean, 3), reshape_global(blood_logvar, 3)),
            "csf": (reshape_global(csf_mean, 2), reshape_global(csf_logvar, 2)),
            "cognition": (reshape_global(cog_mean, 2), reshape_global(cog_logvar, 2)),
            "mri": (reshape_roi(mri_mean), reshape_roi(mri_logvar)),
            "pet": (reshape_roi(pet_mean), reshape_roi(pet_logvar)),
        }

        tn_axis_pooled = torch.cat(
            [pooled[:, :, AXIS_INDEX["T"], :], pooled[:, :, AXIS_INDEX["N"], :]], dim=-1
        )
        diag_logits = self.diag_head(tn_axis_pooled)  # (B, T, 3)
        return out, diag_logits, pooled

    # ---------- forward + loss ----------
    def forward(self, batch: Dict, loss_cfg: dict):
        mean0, logvar0, patient_embed = self.encode_z0(batch)
        z0 = self.reparameterize(mean0, logvar0)
        z_traj = self.integrate(z0, batch["visit_months"], patient_embed)
        decoded, diag_logits, pooled = self.decode(z_traj)

        valid = batch["valid_mask"].float()  # (B,T)
        losses = {}

        # --- reconstruction (Gaussian NLL), 모달리티별 관측 마스크 적용 ---
        recon = 0.0
        for m in ["blood", "csf", "cognition"]:
            mean, logvar = decoded[m]
            target = batch["observations"][m]
            m_mask = batch["obs_mask"][m].float() * valid.unsqueeze(-1)
            nll = 0.5 * (logvar + (target - mean) ** 2 / torch.exp(logvar))
            recon = recon + (nll * m_mask).sum() / m_mask.sum().clamp(min=1.0)

        for m in ["mri", "pet"]:
            mean, logvar = decoded[m]
            target = batch["observations"][m]
            m_mask = batch["obs_mask"][m].float() * valid
            nll = 0.5 * (logvar + (target - mean) ** 2 / torch.exp(logvar))  # (B,T,R)
            m_mask_exp = m_mask.unsqueeze(-1).expand_as(nll)
            recon = recon + (nll * m_mask_exp).sum() / m_mask_exp.sum().clamp(min=1.0)

        losses["recon"] = recon

        # --- KL(q(z0) || N(0, I)) ---
        kl = -0.5 * torch.sum(1 + logvar0 - mean0.pow(2) - logvar0.exp())
        kl = kl / mean0.shape[0]
        losses["kl"] = kl

        # --- identifiability anchor: N축 요약 vs CDRSB 상관 ---
        n_summary = pooled[:, :, AXIS_INDEX["N"], :].mean(dim=-1)  # (B,T)
        cdrsb = batch["observations"]["cognition"][:, :, 0]
        cog_mask = batch["obs_mask"]["cognition"].float() * valid
        anchor_loss = self._masked_neg_corr(n_summary, cdrsb, cog_mask)
        losses["anchor"] = anchor_loss

        # --- ordering: 초반 방문에서 A축이 T축보다 먼저 활성화되어야 함 (heuristic) ---
        a_summary = pooled[:, :, AXIS_INDEX["A"], :].mean(dim=-1)
        t_summary = pooled[:, :, AXIS_INDEX["T"], :].mean(dim=-1)
        T = a_summary.shape[1]
        early = max(1, T // 2)
        margin = 0.0
        ordering_loss = F.relu(t_summary[:, :early] - a_summary[:, :early] + margin)
        ordering_loss = (ordering_loss * valid[:, :early]).sum() / valid[:, :early].sum().clamp(min=1.0)
        losses["ordering"] = ordering_loss

        # --- 보조 진단 분류 ---
        diag_ce = F.cross_entropy(
            diag_logits.reshape(-1, 3), batch["dx"].reshape(-1), reduction="none"
        ).reshape(valid.shape)
        diag_ce = (diag_ce * valid).sum() / valid.sum().clamp(min=1.0)
        losses["diag_ce"] = diag_ce

        total = (
            loss_cfg["recon_weight"] * losses["recon"]
            + loss_cfg["kl_weight"] * losses["kl"]
            + loss_cfg["anchor_weight"] * losses["anchor"]
            + loss_cfg["ordering_weight"] * losses["ordering"]
            + loss_cfg["diag_aux_weight"] * losses["diag_ce"]
        )
        losses["total"] = total
        return total, losses, {"z_traj": z_traj, "diag_logits": diag_logits}

    @staticmethod
    def _masked_neg_corr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        """1 - Pearson corr(x, y) (마스크 적용), 낮을수록 x,y가 잘 정렬됨."""
        mask = mask.bool()
        losses = []
        for b in range(x.shape[0]):
            xb, yb = x[b][mask[b]], y[b][mask[b]]
            if xb.numel() < 2:
                continue
            xb = xb - xb.mean()
            yb = yb - yb.mean()
            denom = (xb.norm() * yb.norm()).clamp(min=1e-6)
            corr = (xb * yb).sum() / denom
            losses.append(1 - corr)
        if not losses:
            return torch.tensor(0.0, device=x.device)
        return torch.stack(losses).mean()
