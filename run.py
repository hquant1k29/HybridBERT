# ============================================================
# RUN PIPELINE - gộp prepare/train/evaluate/visualize vào 1 file
# NOTE: Code được chuyển nguyên khối từ notebook gốc.
# Chỉ thêm import để dùng được sau khi tách các module lõi.
# ============================================================

from src.config import *
from src.graph_builder import *
from src.feature_extractor import *
from src.model import *

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load data
# ============================================================

def load_idx_to_code(jsonl_path: str) -> Dict[str, str]:
    idx_to_code = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx_to_code[str(obj["idx"])] = obj["func"]
    return idx_to_code


def load_pairs_txt(txt_path: str) -> List[Tuple[str, str, int]]:
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 3:
                continue
            idx1, idx2, label = parts
            pairs.append((str(idx1), str(idx2), int(label)))
    return pairs


jsonl_path = os.path.join("data", "data.jsonl")
train_path = os.path.join("data", "train_50000.txt")
test_path = os.path.join("data", "test_50000.txt")
valid_path = os.path.join("data", "valid_50000.txt")

idx_to_code = load_idx_to_code(jsonl_path)
train_pairs = load_pairs_txt(train_path)
valid_pairs = load_pairs_txt(valid_path)
test_pairs  = load_pairs_txt(test_path)

print(len(idx_to_code), len(train_pairs), len(valid_pairs), len(test_pairs))


# ============================================================
# Build vocab
# ============================================================

def build_vocabs_from_pairs(pairs, idx_to_code, limit=None):
    ast_samples, dfg_samples = [], []
    iterable = pairs if limit is None else pairs[:limit]

    for idx1, idx2, _ in iterable:
        if idx1 not in idx_to_code or idx2 not in idx_to_code:
            continue
        for idx in [idx1, idx2]:
            code = idx_to_code[idx]
            try:
                ast_samples.append(build_ast_graph(code, max_nodes=cfg.max_ast_nodes))
                dfg_samples.append(build_dfg_graph(code, max_nodes=cfg.max_dfg_nodes))
            except Exception:
                pass

    build_vocabs(ast_samples, dfg_samples)


build_vocabs_from_pairs(train_pairs, idx_to_code, limit=1000)
print(len(ast_type_vocab), len(dfg_stmt_vocab), len(var_name_vocab))


# ============================================================
# Prepare / load feature cache
# ============================================================


feature_path = os.path.join("data", "precomputed_features_with_stats.pt")


def cache_has_stat_features(path):
    if not os.path.exists(path):
        return False
    try:
        tmp = torch.load(path, map_location="cpu")
        if not isinstance(tmp, dict) or len(tmp) == 0:
            return False
        first_key = next(iter(tmp.keys()))
        first_feat = tmp[first_key]
        return "stat_vec" in first_feat and "stat_meta" in first_feat
    except Exception:
        return False


if not cache_has_stat_features(feature_path):
    print("Đang tính feature + statistical vector....")
    precomputed_features = {}
    
    # Gom tất cả các ID code thực sự được dùng để tránh tính thừa
    used_indices = set()
    for splits in [train_pairs, valid_pairs, test_pairs]:
        for idx1, idx2, _ in splits:
            used_indices.add(idx1)
            used_indices.add(idx2)

    count = 0
    skipped = 0
    total = len(used_indices)
    
    for idx in used_indices:
        if idx in idx_to_code:
            try:
                precomputed_features[idx] = prepare_one_code_sample(idx_to_code[idx])
            except Exception as e:
                skipped += 1
                # Bỏ qua các code bị lỗi cú pháp/parse graph.
                pass
        
        count += 1
        if count % 500 == 0:
            print(f"Đã xử lý {count}/{total} code snippets... skipped={skipped}")

    torch.save(precomputed_features, feature_path)
    print(f"Xong! Đã lưu {feature_path}")
    print(f"Tổng feature: {len(precomputed_features)}, skipped={skipped}")
else:
    print("Đã tìm thấy cache feature có statistical vector, load vào RAM thôi!")
    
precomputed_features = torch.load(feature_path, map_location="cpu")
print(f"Đã load {len(precomputed_features)} features vào RAM sẵn sàng train.")

# Kiểm tra nhanh một sample
if len(precomputed_features) > 0:
    sample_key = next(iter(precomputed_features.keys()))
    print("Sample keys:", precomputed_features[sample_key].keys())
    print("stat_vec shape:", precomputed_features[sample_key]["stat_vec"].shape)


# ============================================================
# Dataset / DataLoader
# ============================================================

def batch_graphs(graph_list, node_key="node_type_ids"):
    node_ids, numeric_feats, edge_index, edge_type_ids, batch_vec = [], [], [], [], []
    node_offset = 0
    
    for i, g in enumerate(graph_list):
        num_nodes = g[node_key].shape[0]
        if num_nodes == 0:
            continue
            
        node_ids.append(g[node_key])
        numeric_feats.append(g["numeric_feats"])
        
        if g["edge_index"].shape[1] > 0:
            edge_index.append(g["edge_index"] + node_offset)
            edge_type_ids.append(g["edge_type_ids"])
            
        batch_vec.append(torch.full((num_nodes,), i, dtype=torch.long))
        node_offset += num_nodes

    return {
        node_key: torch.cat(node_ids, dim=0) if node_ids else torch.tensor([], dtype=torch.long),
        "numeric_feats": torch.cat(numeric_feats, dim=0) if numeric_feats else torch.tensor([], dtype=torch.float),
        "edge_index": torch.cat(edge_index, dim=1) if edge_index else torch.zeros((2, 0), dtype=torch.long),
        "edge_type_ids": torch.cat(edge_type_ids, dim=0) if edge_type_ids else torch.tensor([], dtype=torch.long),
        "batch": torch.cat(batch_vec, dim=0) if batch_vec else torch.tensor([], dtype=torch.long)
    }


class ClonePairDatasetOptimized(Dataset):
    def __init__(self, pairs, precomputed_features):
        self.samples = []
        for idx1, idx2, label in pairs:
            if idx1 in precomputed_features and idx2 in precomputed_features:
                feat1, feat2 = precomputed_features[idx1], precomputed_features[idx2]
                if feat1["ast_graph"]["node_type_ids"].numel() == 0 or feat1["dfg_graph"]["stmt_type_ids"].numel() == 0 or \
                   feat2["ast_graph"]["node_type_ids"].numel() == 0 or feat2["dfg_graph"]["stmt_type_ids"].numel() == 0:
                    continue
                if "stat_vec" not in feat1 or "stat_vec" not in feat2:
                    continue
                self.samples.append((idx1, idx2, label))
        self.features = precomputed_features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        idx1, idx2, label = self.samples[i]
        return {
            "idx1": idx1,
            "idx2": idx2,
            "feat1": self.features[idx1],
            "feat2": self.features[idx2],
            "label": label
        }


def pair_collate_fn_truely_batched(batch):
    if len(batch) == 0:
        return None
    
    feat1_ast = batch_graphs([item["feat1"]["ast_graph"] for item in batch], node_key="node_type_ids")
    feat1_dfg = batch_graphs([item["feat1"]["dfg_graph"] for item in batch], node_key="stmt_type_ids")
    feat1_code = torch.cat([item["feat1"]["code_vec"] for item in batch], dim=0)
    feat1_stat = torch.stack([item["feat1"]["stat_vec"] for item in batch], dim=0)

    feat2_ast = batch_graphs([item["feat2"]["ast_graph"] for item in batch], node_key="node_type_ids")
    feat2_dfg = batch_graphs([item["feat2"]["dfg_graph"] for item in batch], node_key="stmt_type_ids")
    feat2_code = torch.cat([item["feat2"]["code_vec"] for item in batch], dim=0)
    feat2_stat = torch.stack([item["feat2"]["stat_vec"] for item in batch], dim=0)

    # Pair-level similarity/difference features được tạo ngay tại collate vì cần cả feat1 và feat2.
    pair_stat = torch.stack([
        build_pair_stat_from_meta(item["feat1"], item["feat2"])
        for item in batch
    ], dim=0)

    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    idx1_list = [item["idx1"] for item in batch]
    idx2_list = [item["idx2"] for item in batch]
    
    return {
        "idx1": idx1_list,
        "idx2": idx2_list,
        "feat1": {
            "ast_graph": feat1_ast,
            "dfg_graph": feat1_dfg,
            "code_vec": feat1_code,
            "stat_vec": feat1_stat,
            "pair_stat": pair_stat,  # gắn vào feat1 để giữ forward(model(feat1, feat2))
        },
        "feat2": {
            "ast_graph": feat2_ast,
            "dfg_graph": feat2_dfg,
            "code_vec": feat2_code,
            "stat_vec": feat2_stat,
        },
        "label": labels
    }


train_dataset = ClonePairDatasetOptimized(train_pairs, precomputed_features)
valid_dataset = ClonePairDatasetOptimized(valid_pairs, precomputed_features)
test_dataset  = ClonePairDatasetOptimized(test_pairs, precomputed_features)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pair_collate_fn_truely_batched, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pair_collate_fn_truely_batched, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=pair_collate_fn_truely_batched, num_workers=4)

print(len(train_loader))
print(len(valid_loader))
print(len(test_loader))


# ============================================================
# Model / optimizer
# ============================================================

pair_model = PairCloneClassifier(
    ast_vocab_size=len(ast_type_vocab),
    dfg_vocab_size=len(dfg_stmt_vocab),
    hidden_dim=cfg.hidden_dim,
    num_classes=cfg.num_classes,
    stat_dim=STAT_VECTOR_DIM,
    pair_extra_dim=PAIR_EXTRA_DIM,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    [p for p in pair_model.parameters() if p.requires_grad],
    lr=1e-4
)

print(pair_model)
print(device)
print(next(pair_model.parameters()).device)


# ============================================================
# Train / Eval helpers
# ============================================================

def move_feat_to_device(sample_feat, device):
    out = {
        "code_vec": sample_feat["code_vec"].to(device),
        "stat_vec": sample_feat["stat_vec"].to(device),
        "ast_graph": {k: v.to(device) for k, v in sample_feat["ast_graph"].items()},
        "dfg_graph": {k: v.to(device) for k, v in sample_feat["dfg_graph"].items()},
    }
    if "pair_stat" in sample_feat:
        out["pair_stat"] = sample_feat["pair_stat"].to(device)
    return out


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_count = 0.0, 0

    for batch in loader:
        if batch is None:
            continue

        optimizer.zero_grad()
        
        feat1 = move_feat_to_device(batch["feat1"], device)
        feat2 = move_feat_to_device(batch["feat2"], device)
        label = batch["label"].to(device)

        logits = model(feat1, feat2)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        batch_size = label.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            feat1 = move_feat_to_device(batch["feat1"], device)
            feat2 = move_feat_to_device(batch["feat2"], device)
            label = batch["label"].to(device)

            logits = model(feat1, feat2)
            loss = criterion(logits, label)

            batch_size = label.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            pred = torch.argmax(logits, dim=-1)
            y_true.extend(label.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "support_per_class": support_per_class,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_metrics(metrics, class_names=None):
    class_names = class_names or ["Type-1", "Type-2", "Type-3", "Type-4"]
    print(f"loss            : {metrics['loss']:.4f}")
    print(f"accuracy        : {metrics['accuracy']:.4f}")
    print(f"precision_macro : {metrics['precision_macro']:.4f}")
    print(f"recall_macro    : {metrics['recall_macro']:.4f}")
    print(f"f1_macro        : {metrics['f1_macro']:.4f}")
    print()
    for i, name in enumerate(class_names):
        print(
            f"{name:7s} | "
            f"P={metrics['precision_per_class'][i]:.4f}  "
            f"R={metrics['recall_per_class'][i]:.4f}  "
            f"F1={metrics['f1_per_class'][i]:.4f}  "
            f"Support={metrics['support_per_class'][i]}"
        )


def plot_confusion_matrix_percent(metrics, class_names=None, figsize=(7, 7), save_path=None):
    class_names = class_names or ["Type-1", "Type-2", "Type-3", "Type-4"]
    cm = metrics["confusion_matrix"].astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_percent = cm / row_sums * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix (%)',
        ylabel='True label',
        xlabel='Predicted label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_percent.max() / 2.0
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            ax.text(
                j, i, f"{cm_percent[i, j]:.2f}%",
                ha="center", va="center",
                color="white" if cm_percent[i, j] > thresh else "black"
            )

    fig.tight_layout()
    save_path = save_path or os.path.join(OUTPUT_DIR, "confusion_matrix_percent.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")


# ============================================================
# Training block from notebook gốc
# Lưu ý: block này vẫn đang comment đúng như notebook gốc.
# Nếu muốn train thật, bỏ comment phần dưới.
# ============================================================

# Training is enabled below.
start_time = time.time()
best_f1 = -1.0
best_state = None
num_epochs = 30

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_loss = train_one_epoch(pair_model, train_loader, optimizer, criterion, device)
    valid_metrics = eval_one_epoch(pair_model, valid_loader, criterion, device)

    epoch_time = time.time() - epoch_start_time
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
    print(f"train_loss: {train_loss:.4f}")
    print_metrics(valid_metrics)

    print(f"\nTime training epoch: {epoch_time} seconds")
    if valid_metrics["f1_macro"] > best_f1:
        best_f1 = valid_metrics["f1_macro"]
        best_state = {k: v.cpu() for k, v in pair_model.state_dict().items()}
        torch.save(best_state, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print("Saved best model successfully")
        print(f"New best macro F1: {best_f1:.4f}")

total_training_time = time.time() - start_time
print(f"Time training for all epochs: {total_training_time} seconds")
hours = int(total_training_time // 3600)
minutes = int((total_training_time % 3600) // 60)
seconds = int(total_training_time % 60)

print("\n===== TRAINING FINISHED =====")
print(f"Total training time: {hours}h {minutes}m {seconds}s")


# ============================================================
# Load Best Model & Evaluate
# ============================================================

best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(
        f"Missing checkpoint: {best_model_path}. "
        "Place best_model.pth in outputs/ before running evaluation."
    )

best_state = torch.load(best_model_path, map_location=device)
if best_state is not None:
    pair_model.load_state_dict(best_state)
    print("✅ Đã load trọng số hội tụ tốt nhất!")

# Đảm bảo model vẫn ở trên GPU để chạy cho nhanh
pair_model.to(device)

print("\n⏳ Đang chạy đánh giá trên tập Test...")
# start_time = time.time()
test_metrics = eval_one_epoch(pair_model, test_loader, criterion, device)
# total_time = time.time() - start_time//
# print(f"Total time for testing: {total_time}")
nums_sample = len(test_loader.dataset)
# print(f"Avg time for valid test for each sample: {total_time/nums_sample} seconds")

print("\n📊 KẾT QUẢ TẬP TEST:")
print("-" * 50)
print_metrics(test_metrics, class_names=["Type-1", "Type-2", "Type-3", "Type-4"])

print("\n🎨 Đang vẽ Confusion Matrix...")
plot_confusion_matrix_percent(
    test_metrics, 
    class_names=["Type-1", "Type-2", "Type-3", "Type-4"], 
    figsize=(8, 8),
    save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_percent.png")
)


# ============================================================
# t-SNE / ROC
# ============================================================

def collect_outputs_for_tsne_roc(model, loader, device):
    model.eval()

    all_embeddings = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            feat1 = move_feat_to_device(batch["feat1"], device)
            feat2 = move_feat_to_device(batch["feat2"], device)
            labels = batch["label"].to(device)

            logits, embeddings = model(
                feat1,
                feat2,
                return_embedding=True
            )

            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_embeddings.append(embeddings.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_embeddings, all_probs, all_preds, all_labels


embeddings, probs, preds, labels = collect_outputs_for_tsne_roc(
    pair_model,
    test_loader,
    device
)

print("embeddings:", embeddings.shape)
print("probs:", probs.shape)
print("preds:", preds.shape)
print("labels:", labels.shape)


def plot_tsne_all_classes(embeddings, labels, save_path=None):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    emb_2d = tsne.fit_transform(embeddings)

    class_names = ["Type-1", "Type-2", "Type-3", "Type-4"]

    fig, ax = plt.subplots(figsize=(9, 7))

    for c in range(4):
        mask = labels == c
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=10,
            alpha=0.65,
            label=class_names[c]
        )

    ax.set_title("t-SNE Visualization of Pair Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = save_path or os.path.join(OUTPUT_DIR, "tsne_pair_embeddings.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")

    return emb_2d


emb_2d = plot_tsne_all_classes(
    embeddings,
    labels,
    save_path=os.path.join(OUTPUT_DIR, "tsne_pair_embeddings.png")
)


def plot_multiclass_roc(labels, probs, num_classes=4, save_path=None):
    labels_bin = label_binarize(labels, classes=list(range(num_classes)))

    class_names = ["Type-1", "Type-2", "Type-3", "Type-4"]

    fpr = {}
    tpr = {}
    roc_auc = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        ax.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=f"{class_names[i]} AUC = {roc_auc[i]:.4f}"
        )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = save_path or os.path.join(OUTPUT_DIR, "roc_multiclass.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")

    return roc_auc


roc_auc = plot_multiclass_roc(
    labels,
    probs,
    num_classes=4,
    save_path=os.path.join(OUTPUT_DIR, "roc_multiclass.png")
)
print("ROC AUC:", roc_auc)

# emb_2d_34 = plot_tsne_type34(embeddings, labels)
# auc_34 = plot_roc_type3_vs_type4(labels, probs)

sys.exit(0)
