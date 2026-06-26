# ============================================================
# CONFIG / IMPORTS - from notebook cell 3
# NOTE: Code được chuyển nguyên khối từ notebook gốc.
# Chỉ thêm import tương đối ở đầu file để các block chạy được khi tách module.
# ============================================================

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt

import javalang
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

@dataclass
class Config:
    codebert_name: str = "microsoft/codebert-base"
    max_code_length: int = 256
    max_ast_nodes: int = 256
    max_dfg_nodes: int = 128
    hidden_dim: int = 256
    num_classes: int = 4
    dropout: float = 0.2
    freeze_codebert: bool = True

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
