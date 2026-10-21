"""
model.py - Structural-Morphology-Aware Graph Transformer (SGT).

Equation numbers refer to the manuscript (Methods, Section 0.5):

  MorphologicalTokenInit      Eq. (4)      m_i^(0) = W2 ReLU(W1 mu_i + b1) + b2,   mu_i = [d_i, s~_i, c_i]
  MASEPropagation             Eq. (5)      m_i^(l) = LN(W_self m_i^(l-1) + W_agg mean_{j in N(i)} m_j^(l-1))
  GatedCrossStreamFusion      Eqs. (6-7)   g_i = sigma(W_gate [h_i^(l-1) || m_i^(l-1)]),  h_in = h^(l-1) + g ⊙ m^(l-1)
  SpatialRelationshipEncoder  Eq. (8)      phi_ij = 1 / (1 + ||p_i - p_j||^2),  bias = beta * phi_ij
  SREMultiHeadAttention       Eqs. (9-11)  alpha_ij = softmax_{j in N(i)}(q_i^T k_j / sqrt(d_k) + beta phi_ij)
  SGTLayer                    Eqs. (12-15) and Algorithm 1, lines 7-16
  SGT                         Eqs. (2, 16, 17)

Switches (ModelConfig) — defaults implement the manuscript exactly:
  fuse_prev_morph      True : gate/fusion use m^(l-1) (Eqs. 6-7, Alg. 1 lines 9-10).
                       False: gate/fusion use m^(l)   (behaviour of the code before this revision).
  drop_path_per_graph  True : stochastic depth drops a residual branch for a whole graph.
                       False: one Bernoulli draw per NODE (behaviour before this revision).
  use_mase / use_sre   ablations; False removes the morphological stream / the spatial bias.
  variant              "sgt" (default) | "concat": control without MASE in which the three morphological
                       attributes are concatenated to the node features (in_dim + 3 inputs); the SRE is kept.
                       With match_params=True the FFN width is enlarged so that the parameter count matches
                       SGT with the same configuration (see build_model).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add, scatter_mean


# ─── Stochastic depth (DropPath) ─────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """
    Drops a whole residual branch with probability p during training (inverted scaling, identity at test time).

    The input is a flat node tensor (N_total, F) of a mini-batch of graphs. With `batch` given, one Bernoulli
    draw is made per GRAPH, so the branch is dropped for every node of that graph (stochastic depth per sample).
    Without `batch` (or per_graph=False) one draw is made per node.
    """

    def __init__(self, drop_prob: float = 0.0, per_graph: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.per_graph = per_graph

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        if self.per_graph and batch is not None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            mask = torch.empty(num_graphs, dtype=x.dtype, device=x.device).bernoulli_(keep).div_(keep)
            return x * mask[batch].unsqueeze(-1)
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        return x * mask.bernoulli_(keep).div_(keep)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob:.3f}, per_graph={self.per_graph}"


# ─── Eq. (4): morphological token initialisation ─────────────────────────────

class MorphologicalTokenInit(nn.Module):
    """m_i^(0) = W2 · ReLU(W1 · mu_i + b1) + b2,  mu_i = [degree, area, convexity] (3) -> (F)."""

    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, morph: Tensor) -> Tensor:
        return self.mlp(morph)


# ─── Eq. (5): MASE layer-wise propagation ────────────────────────────────────

class MASEPropagation(nn.Module):
    """
    m_i^(l) = LN( W_self m_i^(l-1) + W_agg · (1/|N(i)|) sum_{j in N(i)} m_j^(l-1) )

    W_self, W_agg have no bias and share no parameters with the visual stream. The graph is undirected,
    so aggregating messages src -> dst over all edges gives the mean over N(dst). A node without
    neighbours receives a zero aggregate.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.W_self = nn.Linear(dim, dim, bias=False)
        self.W_agg = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, m: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        agg = scatter_mean(m[src], dst, dim=0, dim_size=m.size(0))
        return self.norm(self.W_self(m) + self.W_agg(agg))


# ─── Eqs. (6-7): gated cross-stream fusion ───────────────────────────────────

class GatedCrossStreamFusion(nn.Module):
    """g = sigma(W_gate [h || m]) in (0,1)^F ;  h_in = h + g ⊙ m.   Returns (h_in, g)."""

    def __init__(self, dim: int):
        super().__init__()
        self.W_gate = nn.Linear(2 * dim, dim)

    def forward(self, h: Tensor, m: Tensor) -> Tuple[Tensor, Tensor]:
        gate = torch.sigmoid(self.W_gate(torch.cat([h, m], dim=-1)))
        return h + gate * m, gate


# ─── Eq. (8): spatial relationship encoder ───────────────────────────────────

class SpatialRelationshipEncoder(nn.Module):
    """
    phi_ij = 1 / (1 + ||p_i - p_j||_2^2) in (0, 1], computed per edge; returns beta * phi_ij.

    p are centroid coordinates normalised to [0, 1] by the longer image side (see dataset.py).
    beta is ONE scalar, initialised to beta_init, shared by all layers and heads.
    """

    def __init__(self, beta_init: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    @staticmethod
    def proximity(pos: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        diff = pos[src] - pos[dst]
        return 1.0 / (1.0 + (diff * diff).sum(dim=-1))

    def forward(self, pos: Tensor, edge_index: Tensor) -> Tensor:
        return self.beta * self.proximity(pos, edge_index)


# ─── Eqs. (9-11): SRE-augmented multi-head attention ─────────────────────────

class SREMultiHeadAttention(nn.Module):
    """
    Sparse attention over the edges of the graph (node i attends to j in N(i); no self-loops):

        alpha_ij = softmax_{j in N(i)}( q_i^T k_j / sqrt(d_k) + beta phi_ij ),   d_k = F / H
        h_att,i  = W_O concat_h( sum_j alpha_ij^h v_j^h )

    The input is already layer-normalised by the caller (Eq. 9 applies LN once). Dropout is applied to
    alpha during training. A node without neighbours receives h_att = W_O·0 = b_O.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.H, self.D = num_heads, dim // num_heads
        self.scale = math.sqrt(self.D)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, spatial_bias: Optional[Tensor],
                return_attn: bool = False):
        N, H, D = x.size(0), self.H, self.D
        Q, K, V = (lin(x).view(N, H, D) for lin in (self.q, self.k, self.v))
        src, dst = edge_index                                       # message j=src -> i=dst
        scores = (Q[dst] * K[src]).sum(-1) / self.scale             # (E, H)
        if spatial_bias is not None:
            scores = scores + spatial_bias.unsqueeze(-1)            # same bias for every head
        alpha = pyg_softmax(scores, dst, num_nodes=N)               # normalised over N(i)
        msgs = (self.drop(alpha).unsqueeze(-1) * V[src]).reshape(-1, H * D)
        out = self.o(scatter_add(msgs, dst, dim=0, dim_size=N))
        return (out, alpha) if return_attn else out


# ─── One SGT layer: Eqs. (12-15), Algorithm 1 lines 7-16 ─────────────────────

class SGTLayer(nn.Module):
    """
        m^(l)    = MASE(m^(l-1))                                               Eq. (12)
        h_in     = h^(l-1) + sigma(W_gate [h^(l-1) || m~]) ⊙ m~                 Eq. (13)
                   m~ = m^(l-1) (manuscript, fuse_prev_morph=True) or m^(l)
        h_att    = MHA_SRE(LN(h_in), beta phi)                                 Eq. (14)
        r        = h_in + DropPath(h_att)
        h^(l)    = r + DropPath(FFN(LN(r)))                                    Eq. (15)
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1,
                 drop_path: float = 0.0, use_mase: bool = True, fuse_prev_morph: bool = True,
                 drop_path_per_graph: bool = True):
        super().__init__()
        self.use_mase, self.fuse_prev_morph = use_mase, fuse_prev_morph
        if use_mase:
            self.mase = MASEPropagation(dim)
            self.fusion = GatedCrossStreamFusion(dim)
        self.attn = SREMultiHeadAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path1 = StochasticDepth(drop_path, drop_path_per_graph)
        self.drop_path2 = StochasticDepth(drop_path, drop_path_per_graph)

    def forward(self, h: Tensor, m: Optional[Tensor], edge_index: Tensor, spatial_bias: Optional[Tensor],
                batch: Optional[Tensor] = None, return_internals: bool = False):
        gate, m_new = None, m
        if self.use_mase:
            m_new = self.mase(m, edge_index)                          # Alg. 1 line 8
            m_fuse = m if self.fuse_prev_morph else m_new
            h_in, gate = self.fusion(h, m_fuse)                       # Alg. 1 lines 9-10
        else:
            h_in = h
        h_att, alpha = self.attn(self.norm1(h_in), edge_index, spatial_bias, return_attn=True)
        r = h_in + self.drop_path1(h_att, batch)                      # Alg. 1 line 15
        h_out = r + self.drop_path2(self.ffn(self.norm2(r)), batch)   # Alg. 1 line 16
        if return_internals:
            return h_out, m_new, {"attention": alpha, "gate": gate}
        return h_out, m_new


# ─── Full model ──────────────────────────────────────────────────────────────

class SGT(nn.Module):
    """
    h^(0) = LN(W_in x)                        Eq. (2)
    m^(0) = MLP(mu)                           Eq. (4)
    beta*phi pre-computed once per forward    Eq. (8), reused by all L layers
    L x SGTLayer                              Eqs. (12-15)
    h_graph = mean_i h_i^(L)                  Eq. (16)
    y = MLP(LN(h_graph))                      Eq. (17)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 6,
                 morph_hidden_dim: int = 64, ffn_dim: int = 512, num_classes: int = 7, dropout: float = 0.1,
                 beta_init: float = 1.0, classifier_hidden: int = 128, drop_path_max: float = 0.15,
                 use_mase: bool = True, use_sre: bool = True, fuse_prev_morph: bool = True,
                 drop_path_per_graph: bool = True, variant: str = "sgt"):
        super().__init__()
        if variant not in ("sgt", "concat"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        if variant == "concat":
            use_mase = False                                       # morphology enters only through the input
        self.use_mase, self.use_sre = use_mase, use_sre
        proj_in = in_dim + 3 if variant == "concat" else in_dim
        self.input_proj = nn.Sequential(nn.Linear(proj_in, hidden_dim), nn.LayerNorm(hidden_dim))
        if use_mase:
            self.morph_init = MorphologicalTokenInit(3, morph_hidden_dim, hidden_dim)
        if use_sre:
            self.sre = SpatialRelationshipEncoder(beta_init)
        # stochastic depth: linearly increasing from 0 (first layer) to drop_path_max (last layer)
        rates = [drop_path_max * i / max(num_layers - 1, 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            SGTLayer(hidden_dim, num_heads, ffn_dim, dropout, rates[i], use_mase, fuse_prev_morph,
                     drop_path_per_graph) for i in range(num_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classifier_hidden),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(classifier_hidden, num_classes))
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

    def _run(self, data: Batch, return_internals: bool = False):
        edge_index, batch = data.edge_index, data.batch
        if batch is None:                                           # single un-batched graph
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        x = torch.cat([data.x, data.morph], dim=-1) if self.variant == "concat" else data.x
        h = self.input_proj(x)
        m = self.morph_init(data.morph) if self.use_mase else None
        bias = self.sre(data.pos, edge_index) if self.use_sre else None
        attn, gates = [], []
        for layer in self.layers:
            if return_internals:
                h, m, aux = layer(h, m, edge_index, bias, batch, return_internals=True)
                attn.append(aux["attention"]); gates.append(aux["gate"])
            else:
                h, m = layer(h, m, edge_index, bias, batch)
        logits = self.classifier(scatter_mean(h, batch, dim=0))
        return logits, attn, gates

    def forward(self, data: Batch) -> Tensor:
        return self._run(data)[0]

    @torch.no_grad()
    def get_attention_and_gates(self, data: Batch) -> Dict:
        """Same computation as forward (eval mode) plus per-layer attention (E, H) and gates (N, F)."""
        was_training = self.training
        self.eval()
        logits, attn, gates = self._run(data, return_internals=True)
        self.train(was_training)
        return {"logits": logits, "attention_weights": attn, "gate_values": gates,
                "beta": float(self.sre.beta) if self.use_sre else None}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Factory ─────────────────────────────────────────────────────────────────

def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sgt_kwargs(cfg) -> dict:
    g = lambda name, default: getattr(cfg, name, default)
    return dict(in_dim=cfg.in_dim, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                morph_hidden_dim=cfg.morph_hidden_dim, ffn_dim=cfg.ffn_dim, num_classes=cfg.num_classes,
                dropout=cfg.dropout, beta_init=cfg.beta_init, classifier_hidden=cfg.classifier_hidden_dim,
                drop_path_max=g("drop_path_max", 0.15), use_mase=g("use_mase", True), use_sre=g("use_sre", True),
                fuse_prev_morph=g("fuse_prev_morph", True), drop_path_per_graph=g("drop_path_per_graph", True),
                variant=g("variant", "sgt"))


def matched_ffn_dim(cfg) -> int:
    """FFN width at which the concat variant has (nearly) the parameter count of SGT with the same config.

    Every extra FFN unit adds 2*hidden_dim + 1 parameters per layer, so the width is solved in closed form
    from the parameter counts of the two models at the configured width.
    """
    kw = _sgt_kwargs(cfg)
    target = _count(SGT(**{**kw, "variant": "sgt", "use_mase": True}))
    base = _count(SGT(**{**kw, "variant": "concat"}))
    per_unit = (2 * cfg.hidden_dim + 1) * cfg.num_layers
    return cfg.ffn_dim + max(0, round((target - base) / per_unit))


def build_model(cfg) -> SGT:
    """Build SGT (or the concat control) from a ModelConfig. Fields missing in old configs take defaults."""
    kw = _sgt_kwargs(cfg)
    if kw["variant"] == "concat" and getattr(cfg, "match_params", True):
        kw["ffn_dim"] = matched_ffn_dim(cfg)
    return SGT(**kw)


def infer_config_from_checkpoint(state_dict: dict, cfg) -> None:
    """Set the architecture fields of cfg IN PLACE from the tensor shapes stored in a checkpoint."""
    sd = state_dict
    if "input_proj.0.weight" in sd:
        cfg.hidden_dim, cfg.in_dim = (int(s) for s in sd["input_proj.0.weight"].shape)
    if "classifier.1.weight" in sd:
        cfg.classifier_hidden_dim, cfg.hidden_dim = (int(s) for s in sd["classifier.1.weight"].shape)
    if "classifier.4.weight" in sd:
        cfg.num_classes = int(sd["classifier.4.weight"].shape[0])
    layer_idx = [int(k.split(".")[1]) for k in sd if k.startswith("layers.") and k.endswith(".norm1.weight")]
    if layer_idx:
        cfg.num_layers = max(layer_idx) + 1
    if "layers.0.ffn.0.weight" in sd:
        cfg.ffn_dim = int(sd["layers.0.ffn.0.weight"].shape[0])
    if "morph_init.mlp.0.weight" in sd:
        cfg.morph_hidden_dim = int(sd["morph_init.mlp.0.weight"].shape[0])
    cfg.use_mase = "morph_init.mlp.0.weight" in sd
    cfg.variant = "sgt"
    if not cfg.use_mase and cfg.in_dim % 2 == 1:                 # concat control: in_dim + 3 inputs
        cfg.variant, cfg.in_dim, cfg.match_params = "concat", cfg.in_dim - 3, False
    cfg.use_sre = "sre.beta" in sd
    while cfg.hidden_dim % cfg.num_heads != 0 and cfg.num_heads > 1:
        cfg.num_heads //= 2
    # NOTE: fuse_prev_morph, drop_path_per_graph and the data settings cannot be read from tensor shapes.
    # Checkpoints trained before this revision need:
    #   model.fuse_prev_morph=false model.drop_path_per_graph=false data.bracs_pos=mm data.bracs_area=zlog
