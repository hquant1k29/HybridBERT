import os
import time
import json
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data  # Batch imported in original code is not used.
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------
# Utility Functions for PDG Creation (CPG for Java)
# -------------------------------------------
def assign_node_type_cpg_java(node):
    # AST nodes for Java components
    if node.strip().startswith(('public ', 'private ', 'protected ', 'static ', 'void ', 'int ', 'String ', 'class ')):
        return 'AST_function_definition'
    elif '(' in node and ')' in node and '{' not in node:
        return 'AST_function_call'
    elif 'if' in node or 'while' in node or 'for' in node or 'switch' in node:
        return 'CFG_control_flow'
    elif '=' in node:
        return 'PDG_data_flow'
    return 'AST_statement'

def assign_edge_type_cpg_java(edge, control_flow_only=False):
    if isinstance(edge, str):
        source_line = edge
    elif isinstance(edge, tuple) and len(edge) == 2:
        source_line = edge[0]
    else:
        return 'AST'
    if any(keyword in source_line for keyword in ['if', 'for', 'while', 'switch']):
        return 'CFG_control_flow'
    if '=' in source_line:
        return 'PDG_data_flow'
    if control_flow_only:
        return 'CFG_control_flow'
    return 'AST'

def generate_cpg_java(code):
    try:
        G = nx.MultiDiGraph()
        lines = code.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            node_type = assign_node_type_cpg_java(line)
            if node_type != 'meaningless':
                G.add_node(i, label=line, type=node_type)
                if i > 0:
                    prev_line = lines[i-1].strip()
                    edge_type = assign_edge_type_cpg_java((prev_line, line))
                    G.add_edge(i-1, i, type=edge_type)
        isolated_nodes = [node for node, degree in G.degree if degree == 0]
        G.remove_nodes_from(isolated_nodes)
        if G.number_of_nodes() == 0:
            return None
        for node in G.nodes(data=True):
            if 'type' not in node[1]:
                node[1]['type'] = 'AST_statement'
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)
    except Exception as e:
        return None

def convert_nx_to_torch_data_cpg(G):
    try:
        node_features = torch.tensor(
            [[int(node_data['type'] == t) for t in ['AST_function_definition', 'AST_function_call', 'CFG_control_flow', 'PDG_data_flow', 'AST_statement']]
             for _, node_data in G.nodes(data=True)],
            dtype=torch.float
        )
        edge_list = list(G.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index)
    except Exception as e:
        print(f"Error in convert_nx_to_torch_data_cpg: {str(e)}")
        return None

def create_cpg_pairs(data_file, jsonl_file):
    try:
        with open(jsonl_file, 'r') as f:
            url_to_code = {entry['idx']: entry['func'] for entry in (json.loads(line.strip()) for line in f)}
    except Exception as e:
        print(f"Error reading JSONL file: {str(e)}")
        return [], []
    cpg_pairs, true_labels = [], []
    try:
        with open(data_file, 'r') as f:
            for line in tqdm(f, desc="Creating PDG pairs"):
                try:
                    url1, url2, label = line.strip().split('\t')
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                try:
                    cpg1, cpg2 = generate_cpg_java(url_to_code[url1]), generate_cpg_java(url_to_code[url2])
                    if cpg1 is None or cpg2 is None:
                        continue
                    cpg_pairs.append((convert_nx_to_torch_data_cpg(cpg1), convert_nx_to_torch_data_cpg(cpg2)))
                    true_labels.append(int(label) if label in {'1', '2', '3'} else 0)
                except Exception as e:
                    print(f"Error processing pair {url1} and {url2}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        return [], []
    print(f"Created {len(cpg_pairs)} pairs from {data_file}")
    return cpg_pairs, true_labels

# -------------------------------------------
# Model Definitions
# -------------------------------------------
# Enhanced GNN Layer: Combines a GINConv and a two-stage GATConv then concatenates their outputs.
class EnhancedGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedGNNLayer, self).__init__()
        self.gin_conv = GINConv(nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        ))
        self.gat_conv1 = GATConv(input_dim, output_dim // 4, heads=4, concat=True)
        self.gat_conv2 = GATConv((output_dim // 4) * 4, output_dim // 2, heads=2, concat=True)
    
    def forward(self, x, edge_index):
        x1 = self.gin_conv(x, edge_index)
        x2 = self.gat_conv1(x, edge_index)
        x3 = self.gat_conv2(x2, edge_index)
        return torch.cat([x1, x3], dim=-1)

# Enhanced GNN: Stacks several EnhancedGNNLayers with batch normalization and dropout.
class EnhancedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3):
        super(EnhancedGNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        current_dim = input_dim
        for _ in range(num_layers - 1):
            self.conv_layers.append(EnhancedGNNLayer(current_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 2))
            current_dim = hidden_dim * 2
        self.conv_layers.append(EnhancedGNNLayer(current_dim, output_dim))
        self.batch_norms.append(nn.BatchNorm1d(output_dim * 2))
    
    def forward(self, x, edge_index):
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x

# Enhanced Graph Matcher: Pools using both global mean and add pooling, concatenates them, and processes through an MLP.
class EnhancedGraphMatcher(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gembd_vec_dim=128):
        super(EnhancedGraphMatcher, self).__init__()
        self.gnn = EnhancedGNN(input_dim, hidden_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim * 8, gembd_vec_dim),  # output_dim * 8 because we concatenate 4 pooling operations
            nn.ReLU(),
            nn.Linear(gembd_vec_dim, gembd_vec_dim),
            nn.ReLU(),
            nn.Linear(gembd_vec_dim, gembd_vec_dim // 2)
        )
    
    def forward(self, data1, data2):
        h1 = self.gnn(data1.x, data1.edge_index)
        h2 = self.gnn(data2.x, data2.edge_index)
        h1_mean = global_mean_pool(h1, data1.batch if hasattr(data1, 'batch') else torch.zeros(h1.size(0), dtype=torch.long, device=device))
        h2_mean = global_mean_pool(h2, data2.batch if hasattr(data2, 'batch') else torch.zeros(h2.size(0), dtype=torch.long, device=device))
        h1_sum = global_add_pool(h1, data1.batch if hasattr(data1, 'batch') else torch.zeros(h1.size(0), dtype=torch.long, device=device))
        h2_sum = global_add_pool(h2, data2.batch if hasattr(data2, 'batch') else torch.zeros(h2.size(0), dtype=torch.long, device=device))
        h_combined = torch.cat((h1_mean, h2_mean, h1_sum, h2_sum), dim=1)
        return self.mlp(h_combined)

# Enhanced Clone Detector: Uses the EnhancedGraphMatcher and then a classifier.
class EnhancedCloneDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gembd_vec_dim=128):
        super(EnhancedCloneDetector, self).__init__()
        self.graph_matcher = EnhancedGraphMatcher(input_dim, hidden_dim, output_dim, gembd_vec_dim)
        self.classifier = nn.Sequential(
            nn.Linear(gembd_vec_dim // 2, gembd_vec_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gembd_vec_dim // 4, 4)
        )
    
    def forward(self, data1, data2):
        out = self.graph_matcher(data1, data2)
        logits = self.classifier(out)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return logits, probabilities, predicted_class

# -------------------------------------------
# Checkpoint, Training, and Evaluation Functions
# -------------------------------------------
# Save only the best model checkpoint to a fixed file ("best_model.pth")
def save_best_checkpoint(model, optimizer, epoch, f1_score, accuracy, precision, recall, metrics, output_dir):
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': {
            'best_f1': f1_score,
            'best_accuracy': accuracy,
            'best_precision': precision,
            'best_recall': recall,
            'per_type_metrics': metrics
        }
    }, checkpoint_path)
    print(f"Best model checkpoint saved to {checkpoint_path}")

# Training function (saves only the best model)
def train(model, train_loader, valid_loader, epochs=100, output_dir="checkpoints"):
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            graph_pair, labels = batch
            data1, data2 = graph_pair
            optimizer.zero_grad()
            logits, _, _ = model(data1.to(device), data2.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        val_loss, val_metrics, _, _, _ = evaluate(model, valid_loader)
        val_f1 = val_metrics['overall']['f1']
        val_accuracy = val_metrics['overall']['accuracy']
        val_precision = val_metrics['overall']['precision']
        val_recall = val_metrics['overall']['recall']
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_best_checkpoint(model, optimizer, epoch, best_f1, val_accuracy, val_precision, val_recall, val_metrics, output_dir)

# Evaluation function that also measures average inference time per sample
def evaluate(model, data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    all_preds, all_labels = [], []
    total_samples = 0
    inference_time = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            graph_pair, labels = batch
            data1, data2 = graph_pair
            start = time.time()
            logits, _, preds = model(data1.to(device), data2.to(device))
            end = time.time()
            inference_time += (end - start)
            loss = criterion(logits, labels.to(device))
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)
    avg_inference_time = inference_time / total_samples if total_samples > 0 else None
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    overall_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }
    metrics = {'overall': overall_metrics}
    for clone_type in range(4):
        type_preds, type_labels = (all_preds == clone_type), (all_labels == clone_type)
        metrics[f'Type-{clone_type+1}'] = {
            'accuracy': accuracy_score(type_labels, type_preds),
            'precision': precision_score(type_labels, type_preds, zero_division=0),
            'recall': recall_score(type_labels, type_preds, zero_division=0),
            'f1': f1_score(type_labels, type_preds, zero_division=0)
        }
    avg_loss = total_loss / len(data_loader)
    return avg_loss, metrics, all_preds, all_labels, avg_inference_time

# -------------------------------------------
# Main Execution Logic (with Exact Dataset Splits)
# -------------------------------------------
num_sample = 1000000
num_epochs = 100
train_file = f'dataset/Multiclass/train_{num_sample}.txt'
val_file   = f'dataset/Multiclass/valid_{num_sample}.txt'
test_file  = f'dataset/Multiclass/test_{num_sample}.txt'
jsonl_file = 'dataset/Multiclass/data.jsonl'

train_pdg_pairs, train_labels = create_cpg_pairs(train_file, jsonl_file)
val_pdg_pairs, val_labels     = create_cpg_pairs(val_file, jsonl_file)
test_pdg_pairs, test_labels   = create_cpg_pairs(test_file, jsonl_file)

train_loader = DataLoader(list(zip(train_pdg_pairs, train_labels)), batch_size=64, shuffle=True)
val_loader   = DataLoader(list(zip(val_pdg_pairs, val_labels)), batch_size=64)
test_loader  = DataLoader(list(zip(test_pdg_pairs, test_labels)), batch_size=64)

input_dim = 5
hidden_dim = 128
output_dim = 256
num_layers = 4
dropout_rate = 0.01
model = EnhancedCloneDetector(input_dim, hidden_dim, output_dim).to(device)

output_dir = f'checkpoints_{num_epochs}_epoch/CPG/{num_sample}_sample'
os.makedirs(output_dir, exist_ok=True)

# Train the model (only best model is saved)
train(model, train_loader, val_loader, epochs=num_epochs, output_dir=output_dir)

# Automatically load the best model for final testing
best_model_path = os.path.join(output_dir, "best_model.pth")
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nLoaded best model from {best_model_path}")

# Evaluate the best model on the test set (including average inference time per sample)
test_loss, test_metrics, all_preds, all_labels, avg_inference_time = evaluate(model, test_loader)
print("\nTest Evaluation Results:")
print(f"Loss: {test_loss:.4f}")
overall = test_metrics['overall']
print("Overall metrics:")
print(f"  Accuracy: {overall['accuracy']:.4f}")
print(f"  Precision: {overall['precision']:.4f}")
print(f"  Recall: {overall['recall']:.4f}")
print(f"  F1: {overall['f1']:.4f}")
print(f"Average inference time per sample: {avg_inference_time:.6f} seconds")
for key, metrics in test_metrics.items():
    if key != 'overall':
        print(f"\n{key}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")

# Save test evaluation results as JSON for later review
eval_results = {
    'loss': test_loss,
    'metrics': test_metrics,
    'avg_inference_time_per_sample': avg_inference_time
}
json_path = os.path.join(output_dir, "test_evaluation_results.json")
with open(json_path, 'w') as f:
    json.dump(eval_results, f, indent=4)
print(f"\nTest evaluation results saved to {json_path}")

# Draw and save a normalized confusion matrix (display percentages) as an SVG file
cm = confusion_matrix(all_labels, all_preds, normalize='true')
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
classes = [f"Type-{i+1}" for i in range(4)]
tick_marks = np.arange(len(classes))
ax.set(xticks=tick_marks, yticks=tick_marks, xticklabels=classes, yticklabels=classes,
       title='Normalized Confusion Matrix', ylabel='True label', xlabel='Predicted label')
fmt = '.1%'  # Format values as percentages
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
cm_svg_path = os.path.join(output_dir, "confusion_matrix.svg")
plt.savefig(cm_svg_path, format='svg')
plt.close(fig)
print(f"Confusion matrix saved to {cm_svg_path}")
