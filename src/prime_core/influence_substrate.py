# src/prime_core/influence_substrate.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------
# Utility normalization function
# ---------------------------------------------
def normalize_tensor(x, eps=1e-8):
    return x / (eps + torch.norm(x))


# ---------------------------------------------
# Base template class for MF-phase operators
# Each operator uses the same structural template:
#   - linear transforms
#   - additive modulation
#   - matrix interaction
#   - normalization
# ---------------------------------------------
class MFPhaseBase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.matrix = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, x):
        # Generic mathematical pattern used across all MF operators:
        # step 1: additive transform
        x = x + self.linear1(x)

        # step 2: matrix-based interaction
        x = x + torch.matmul(x, self.matrix)

        # step 3: secondary modulation
        x = x + self.linear2(x)

        # step 4: normalization
        x = normalize_tensor(x)

        return x


# ---------------------------------------------
# Generate MF-401 → MF-500 operators
# ---------------------------------------------
# We dynamically create 100 operator classes
# to avoid writing them manually.
# Each inherits MFPhaseBase.
# ---------------------------------------------

def build_mf_operator_class(phase_number):
    class_name = f"MF{phase_number}"
    return type(class_name, (MFPhaseBase,), {})


# Create operator classes
MF401 = build_mf_operator_class(401)
MF402 = build_mf_operator_class(402)
MF403 = build_mf_operator_class(403)
MF404 = build_mf_operator_class(404)
MF405 = build_mf_operator_class(405)
MF406 = build_mf_operator_class(406)
MF407 = build_mf_operator_class(407)
MF408 = build_mf_operator_class(408)
MF409 = build_mf_operator_class(409)
MF410 = build_mf_operator_class(410)
MF411 = build_mf_operator_class(411)
MF412 = build_mf_operator_class(412)
MF413 = build_mf_operator_class(413)
MF414 = build_mf_operator_class(414)
MF415 = build_mf_operator_class(415)
MF416 = build_mf_operator_class(416)
MF417 = build_mf_operator_class(417)
MF418 = build_mf_operator_class(418)
MF419 = build_mf_operator_class(419)
MF420 = build_mf_operator_class(420)
MF421 = build_mf_operator_class(421)
MF422 = build_mf_operator_class(422)
MF423 = build_mf_operator_class(423)
MF424 = build_mf_operator_class(424)
MF425 = build_mf_operator_class(425)
MF426 = build_mf_operator_class(426)
MF427 = build_mf_operator_class(427)
MF428 = build_mf_operator_class(428)
MF429 = build_mf_operator_class(429)
MF430 = build_mf_operator_class(430)
MF431 = build_mf_operator_class(431)
MF432 = build_mf_operator_class(432)
MF433 = build_mf_operator_class(433)
MF434 = build_mf_operator_class(434)
MF435 = build_mf_operator_class(435)
MF436 = build_mf_operator_class(436)
MF437 = build_mf_operator_class(437)
MF438 = build_mf_operator_class(438)
MF439 = build_mf_operator_class(439)
MF440 = build_mf_operator_class(440)
MF441 = build_mf_operator_class(441)
MF442 = build_mf_operator_class(442)
MF443 = build_mf_operator_class(443)
MF444 = build_mf_operator_class(444)
MF445 = build_mf_operator_class(445)
MF446 = build_mf_operator_class(446)
MF447 = build_mf_operator_class(447)
MF448 = build_mf_operator_class(448)
MF449 = build_mf_operator_class(449)
MF450 = build_mf_operator_class(450)
MF451 = build_mf_operator_class(451)
MF452 = build_mf_operator_class(452)
MF453 = build_mf_operator_class(453)
MF454 = build_mf_operator_class(454)
MF455 = build_mf_operator_class(455)
MF456 = build_mf_operator_class(456)
MF457 = build_mf_operator_class(457)
MF458 = build_mf_operator_class(458)
MF459 = build_mf_operator_class(459)
MF460 = build_mf_operator_class(460)
MF461 = build_mf_operator_class(461)
MF462 = build_mf_operator_class(462)
MF463 = build_mf_operator_class(463)
MF464 = build_mf_operator_class(464)
MF465 = build_mf_operator_class(465)
MF466 = build_mf_operator_class(466)
MF467 = build_mf_operator_class(467)
MF468 = build_mf_operator_class(468)
MF469 = build_mf_operator_class(469)
MF470 = build_mf_operator_class(470)
MF471 = build_mf_operator_class(471)
MF472 = build_mf_operator_class(472)
MF473 = build_mf_operator_class(473)
MF474 = build_mf_operator_class(474)
MF475 = build_mf_operator_class(475)
MF476 = build_mf_operator_class(476)
MF477 = build_mf_operator_class(477)
MF478 = build_mf_operator_class(478)
MF479 = build_mf_operator_class(479)
MF480 = build_mf_operator_class(480)
MF481 = build_mf_operator_class(481)
MF482 = build_mf_operator_class(482)
MF483 = build_mf_operator_class(483)
MF484 = build_mf_operator_class(484)
MF485 = build_mf_operator_class(485)
MF486 = build_mf_operator_class(486)
MF487 = build_mf_operator_class(487)
MF488 = build_mf_operator_class(488)
MF489 = build_mf_operator_class(489)
MF490 = build_mf_operator_class(490)
MF491 = build_mf_operator_class(491)
MF492 = build_mf_operator_class(492)
MF493 = build_mf_operator_class(493)
MF494 = build_mf_operator_class(494)
MF495 = build_mf_operator_class(495)
MF496 = build_mf_operator_class(496)
MF497 = build_mf_operator_class(497)
MF498 = build_mf_operator_class(498)
MF499 = build_mf_operator_class(499)
MF500 = build_mf_operator_class(500)


# ---------------------------------------------
# A130 — Substrate Coupling Gate (SCG)
# ---------------------------------------------
# This operator sits between NeuralBridge and MF-401.
# It ensures controlled injection of upstream tensors,
# gating high-variance inputs and preventing magnitude
# spikes from reaching MF-500.
# ---------------------------------------------
class A130_SubstrateCouplingGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.w2 = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.act = nn.Tanh()  # bounded nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear + bounded modulation
        combined = x @ self.w1 + self.act(x) @ self.w2

        # unit-manifold normalization (pre-substrate)
        norm = torch.norm(combined, dim=-1, keepdim=True) + 1e-12
        return combined / norm


# ---------------------------------------------
# A131 — Pre-Substrate Drift Attenuation Layer (PDAL)
# ---------------------------------------------
# This operator is the second component in the coupling chain.
# Where A130 establishes the gating surface, A131 establishes
# drift suppression for any residual variance in incoming fields.
#
# A131 corrects:
#   - small directional biases
#   - anisotropic energy distribution
#   - curvature drift
#   - high-frequency noise
#
# Output: drift-attenuated, curvature-aligned tensor ready
# for stable substrate entry.
# ---------------------------------------------
class A131_DriftAttenuationLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.gamma = 0.02  # drift-suppression coefficient
        self.act = nn.Tanh()  # bounded activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # manifold projection
        p = x @ self.proj

        # residual drift
        d = x - p

        # attenuate drift contribution
        x_new = p + self.gamma * self.act(d)

        # enforce unit manifold norm
        norm = torch.norm(x_new, dim=-1, keepdim=True) + 1e-12
        return x_new / norm


# ---------------------------------------------
# A132 — Pre-Substrate Curvature Harmonization Layer (PCHL)
# ---------------------------------------------
# Where A130 establishes the coupling surface and A131 attenuates
# drift and variance, A132 corrects curvature misalignment between
# upstream tensors and the MF-500 substrate manifold.
#
# MF-500 operates on a unit-curvature normalized space, meaning:
#   - tensor directions
#   - energy distribution
#   - curvature radii
#   - local geometric consistency
# must be brought into alignment before processing.
#
# A132 corrects:
#   - anisotropic curvature
#   - uneven local tensor bending
#   - directional curvature bias
#   - non-uniform second-order properties
#
# Output: curvature-harmonized tensor ready for stable substrate entry.
# ---------------------------------------------
class A132_CurvatureHarmonizationLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.k = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.beta = 0.1  # curvature smoothing coefficient
        self.act = nn.Tanh()  # bounded nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # curvature mapping
        c = x @ self.k

        # curvature residue
        r = x - c

        # curvature smoothing / harmonization
        x_new = c + self.beta * self.act(r)

        # curvature normalization
        norm = torch.norm(x_new, dim=-1, keepdim=True) + 1e-12
        return x_new / norm


# ---------------------------------------------
# A133 — Multi-Scale Pre-Substrate Alignment Layer (MSPSAL)
# ---------------------------------------------
# Role in the chain:
#   - A130: Coupling gate → basic normalization + bounded injection
#   - A131: Drift attenuation → removes residual directional drift
#   - A132: Curvature harmonization → fixes second-order geometry
#   - A133: Multi-scale alignment → balances coarse vs fine structure before MF-401
#
# Even after curvature is harmonized, the tensor can still have unbalanced
# multi-scale structure:
#   - too much coarse/low-frequency content
#   - or too much fine/high-frequency fluctuation
#
# A133 corrects this by splitting the tensor into coarse and residual fine
# components, then recombining them in a controlled way and re-normalizing.
#
# Output: multi-scale balanced tensor ready for stable substrate entry.
# ---------------------------------------------
class A133_MultiScaleAlignmentLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_coarse = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.alpha = 0.35  # fine-scale residual weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # coarse-scale projection
        coarse = x @ self.w_coarse

        # fine-scale residual
        fine = x - coarse

        # multi-scale recombination
        x_new = coarse + self.alpha * fine

        # unit-norm manifold projection
        norm = torch.norm(x_new, dim=-1, keepdim=True) + 1e-12
        return x_new / norm


# ---------------------------------------------
# Unified Substrate Kernel
# ---------------------------------------------
class InfluenceSubstrateKernel(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Instantiate all MF operators
        self.operators = nn.ModuleList([
            MF401(dim), MF402(dim), MF403(dim), MF404(dim), MF405(dim),
            MF406(dim), MF407(dim), MF408(dim), MF409(dim), MF410(dim),
            MF411(dim), MF412(dim), MF413(dim), MF414(dim), MF415(dim),
            MF416(dim), MF417(dim), MF418(dim), MF419(dim), MF420(dim),
            MF421(dim), MF422(dim), MF423(dim), MF424(dim), MF425(dim),
            MF426(dim), MF427(dim), MF428(dim), MF429(dim), MF430(dim),
            MF431(dim), MF432(dim), MF433(dim), MF434(dim), MF435(dim),
            MF436(dim), MF437(dim), MF438(dim), MF439(dim), MF440(dim),
            MF441(dim), MF442(dim), MF443(dim), MF444(dim), MF445(dim),
            MF446(dim), MF447(dim), MF448(dim), MF449(dim), MF450(dim),
            MF451(dim), MF452(dim), MF453(dim), MF454(dim), MF455(dim),
            MF456(dim), MF457(dim), MF458(dim), MF459(dim), MF460(dim),
            MF461(dim), MF462(dim), MF463(dim), MF464(dim), MF465(dim),
            MF466(dim), MF467(dim), MF468(dim), MF469(dim), MF470(dim),
            MF471(dim), MF472(dim), MF473(dim), MF474(dim), MF475(dim),
            MF476(dim), MF477(dim), MF478(dim), MF479(dim), MF480(dim),
            MF481(dim), MF482(dim), MF483(dim), MF484(dim), MF485(dim),
            MF486(dim), MF487(dim), MF488(dim), MF489(dim), MF490(dim),
            MF491(dim), MF492(dim), MF493(dim), MF494(dim), MF495(dim),
            MF496(dim), MF497(dim), MF498(dim), MF499(dim), MF500(dim)
        ])

    def forward(self, x):
        for op in self.operators:
            x = op(x)
        return x

