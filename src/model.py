# ============================================================
# MODEL - from notebook cell 12
# NOTE: Code được chuyển nguyên khối từ notebook gốc.
# Chỉ thêm import tương đối ở đầu file để các block chạy được khi tách module.
# ============================================================

from .config import *
from .feature_extractor import STAT_VECTOR_DIM, PAIR_EXTRA_DIM

class ASTEncoder(nn.Module):
    def __init__(self, ast_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.type_emb = nn.Embedding(ast_vocab_size, hidden_dim)
        self.feat_proj = nn.Linear(10, hidden_dim)
        self.gnn1 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, node_type_ids, numeric_feats, edge_index, batch):
        x = self.type_emb(node_type_ids) + self.feat_proj(numeric_feats)
        x = F.relu(self.gnn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gnn2(x, edge_index))
        return global_mean_pool(x, batch)

class DFGEncoder(nn.Module):
    def __init__(self, stmt_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.type_emb = nn.Embedding(stmt_vocab_size, hidden_dim)
        self.feat_proj = nn.Linear(5, hidden_dim)
        self.gnn1 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, stmt_type_ids, numeric_feats, edge_index, batch):
        x = self.type_emb(stmt_type_ids) + self.feat_proj(numeric_feats)
        x = F.relu(self.gnn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gnn2(x, edge_index))
        return global_mean_pool(x, batch)

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.proj_b = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, a, b):
        a_ = self.proj_a(a)
        b_ = self.proj_b(b)
        g = torch.sigmoid(self.gate(torch.cat([a_, b_], dim=-1)))
        return g * a_ + (1 - g) * b_

class BaseEncoder(nn.Module):
    def __init__(self, ast_vocab_size, dfg_vocab_size, hidden_dim=256, codebert_dim=768):
        super().__init__()
        self.ast_encoder = ASTEncoder(ast_vocab_size, hidden_dim)
        self.dfg_encoder = DFGEncoder(dfg_vocab_size, hidden_dim)
        self.code_proj = nn.Linear(codebert_dim, hidden_dim)
        self.struct_fusion = GatedFusion(hidden_dim)
        self.final_fusion = GatedFusion(hidden_dim)

    def forward(self, code_vec, ast_graph, dfg_graph):
        h_code = self.code_proj(code_vec)
        h_ast = self.ast_encoder(
            ast_graph["node_type_ids"], ast_graph["numeric_feats"], ast_graph["edge_index"], ast_graph["batch"]
        )
        h_dfg = self.dfg_encoder(
            dfg_graph["stmt_type_ids"], dfg_graph["numeric_feats"], dfg_graph["edge_index"], dfg_graph["batch"]
        )
        h_struct = self.struct_fusion(h_ast, h_dfg)
        return self.final_fusion(h_code, h_struct)

class PairCloneClassifier(nn.Module):
    def __init__(
        self,
        ast_vocab_size,
        dfg_vocab_size,
        hidden_dim=256,
        num_classes=4,
        codebert_dim=768,
        stat_dim=STAT_VECTOR_DIM,
        pair_extra_dim=PAIR_EXTRA_DIM,
    ):
        super().__init__()
        self.base = BaseEncoder(ast_vocab_size, dfg_vocab_size, hidden_dim, codebert_dim)
        self.stat_dim = stat_dim
        self.pair_extra_dim = pair_extra_dim

        base_pair_dim = hidden_dim * 4
        stat_pair_dim = stat_dim * 4
        total_dim = base_pair_dim + stat_pair_dim + pair_extra_dim

        self.cls = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, feat1, feat2, return_embedding=False):
        h1 = self.base(feat1["code_vec"], feat1["ast_graph"], feat1["dfg_graph"])
        h2 = self.base(feat2["code_vec"], feat2["ast_graph"], feat2["dfg_graph"])

        stat1 = feat1["stat_vec"]
        stat2 = feat2["stat_vec"]
        pair_extra = feat1.get("pair_stat", None)
        if pair_extra is None:
            # Fallback để tránh crash khi debug thủ công.
            pair_extra = torch.zeros((stat1.size(0), self.pair_extra_dim), device=stat1.device, dtype=stat1.dtype)

        z_base = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        z_stat = torch.cat([stat1, stat2, torch.abs(stat1 - stat2), stat1 * stat2], dim=-1)
        z = torch.cat([z_base, z_stat, pair_extra], dim=-1)

        logits = self.cls(z)

        if return_embedding:
            return logits, z

        return logits
        # return self.cls(z)
