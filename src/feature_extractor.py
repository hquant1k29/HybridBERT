# ============================================================
# FEATURE EXTRACTOR - from notebook cells 9,11,15
# NOTE: Code được chuyển nguyên khối từ notebook gốc.
# Chỉ thêm import tương đối ở đầu file để các block chạy được khi tách module.
# ============================================================

from .config import *
from .graph_builder import *

# ============================================================
# Statistical features for Type-3 / Type-4 improvement
# ============================================================
# Ý tưởng:
# - Mỗi code snippet -> stat_vec: thống kê cấu trúc / API / literal / DFG
# - Mỗi pair -> pair_stat: độ giống nhau API/type/literal/variable + độ lệch feature
# - Classifier sẽ concat thêm [stat1, stat2, |stat1-stat2|, stat1*stat2, pair_stat]

import math
import re
from typing import Set

STAT_KEYS = [
    # size / structure
    "num_ast_nodes",
    "max_ast_depth",
    "num_statements",
    "num_if",
    "num_for",
    "num_while",
    "num_do",
    "num_switch",
    "num_try",
    "num_catch",
    "num_finally",
    "num_return",
    "num_throw",
    "num_assignment",
    "num_binary_op",

    # operation categories
    "file_io_count",
    "stream_io_count",
    "buffer_io_count",
    "zip_io_count",
    "url_network_count",
    "xml_dom_count",
    "string_processing_count",
    "hash_digest_count",
    "collection_count",
    "date_time_count",

    # call / type / literal / variable
    "num_method_invocations",
    "num_unique_api_calls",
    "num_unique_types",
    "num_string_literals",
    "num_numeric_literals",
    "num_variables",

    # DFG summary
    "dfg_num_nodes",
    "dfg_num_edges",
    "dfg_density",
]

PAIR_STAT_KEYS = [
    "api_jaccard",
    "type_jaccard",
    "string_literal_jaccard",
    "variable_jaccard",
    "abs_feature_diff_mean",
    "abs_feature_diff_max",
]

STAT_VECTOR_DIM = len(STAT_KEYS)
PAIR_EXTRA_DIM = len(PAIR_STAT_KEYS)

FILE_IO_HINTS = {
    "File", "Files", "Path", "Paths", "RandomAccessFile", "FileChannel",
    "FileInputStream", "FileOutputStream", "FileReader", "FileWriter",
}
STREAM_IO_HINTS = {
    "InputStream", "OutputStream", "Reader", "Writer", "DataInputStream", "DataOutputStream",
    "ObjectInputStream", "ObjectOutputStream", "ByteArrayInputStream", "ByteArrayOutputStream",
}
BUFFER_IO_HINTS = {"BufferedInputStream", "BufferedOutputStream", "BufferedReader", "BufferedWriter", "ByteBuffer"}
ZIP_IO_HINTS = {"ZipInputStream", "ZipOutputStream", "ZipEntry", "JarInputStream", "JarOutputStream", "GZIPInputStream", "GZIPOutputStream"}
URL_NETWORK_HINTS = {"URL", "URI", "URLConnection", "HttpURLConnection", "Socket", "ServerSocket", "InetAddress"}
XML_DOM_HINTS = {"Document", "Element", "Node", "NodeList", "DocumentBuilder", "SAXParser", "XMLReader"}
COLLECTION_HINTS = {"List", "ArrayList", "Map", "HashMap", "Set", "HashSet", "Collection", "Iterator", "Hashtable", "Vector"}
DATE_TIME_HINTS = {"Date", "Calendar", "LocalDate", "LocalDateTime", "Instant", "SimpleDateFormat", "DateFormat"}
HASH_DIGEST_HINTS = {"MessageDigest", "DigestInputStream", "Mac", "SHA", "MD5", "SHA1", "SHA-1", "SHA256", "SHA-256"}

STRING_METHOD_HINTS = {
    "substring", "indexOf", "lastIndexOf", "startsWith", "endsWith", "contains", "replace",
    "replaceAll", "split", "trim", "toLowerCase", "toUpperCase", "charAt", "append", "toString",
}
FILE_METHOD_HINTS = {
    "exists", "mkdir", "mkdirs", "delete", "renameTo", "createNewFile", "listFiles", "isFile",
    "isDirectory", "getAbsolutePath", "getName", "length", "lastModified", "setLastModified",
}
STREAM_METHOD_HINTS = {"read", "write", "close", "flush", "skip", "available", "reset", "mark"}
HASH_METHOD_HINTS = {"digest", "update", "getInstance"}
COLLECTION_METHOD_HINTS = {"add", "remove", "put", "get", "containsKey", "containsValue", "iterator", "hasNext", "next", "size", "isEmpty"}


def _to_plain_set(xs):
    return {str(x) for x in xs if x is not None and str(x) != ""}


def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def get_type_name_from_node(node):
    """Lấy type/class name từ các node khai báo hoặc creator trong javalang."""
    for attr in ["type"]:
        if hasattr(node, attr):
            t = getattr(node, attr)
            if hasattr(t, "name") and isinstance(t.name, str):
                return t.name
    if hasattr(node, "name") and type(node).__name__ in {"ClassCreator", "ReferenceType", "BasicType"}:
        return str(node.name)
    return ""


def extract_code_stats(code: str, max_dfg_nodes: int = None):
    """
    Return:
        stat_vec: Tensor [STAT_VECTOR_DIM]
        meta: các set dùng để tạo pair similarity: api_calls, type_names, string_literals, variables
    """
    stats = {k: 0.0 for k in STAT_KEYS}
    meta = {
        "api_calls": set(),
        "type_names": set(),
        "string_literals": set(),
        "numeric_literals": set(),
        "variables": set(),
    }

    try:
        root = parse_java_method(code)
    except Exception:
        return torch.zeros(STAT_VECTOR_DIM, dtype=torch.float), meta

    # AST-level statistics
    for node in walk_tree(root):
        if not isinstance(node, javalang.ast.Node):
            continue

        name = type(node).__name__
        stats["num_ast_nodes"] += 1

        # depth: ưu tiên field depth nếu graph builder đã có, còn không sẽ tính riêng bên dưới
        # Ở đây tính lại max depth bằng DFS bên dưới cho chắc chắn.

        if is_statement_node(node):
            stats["num_statements"] += 1

        if name == "IfStatement": stats["num_if"] += 1
        elif name == "ForStatement": stats["num_for"] += 1
        elif name == "WhileStatement": stats["num_while"] += 1
        elif name == "DoStatement": stats["num_do"] += 1
        elif name == "SwitchStatement": stats["num_switch"] += 1
        elif name == "TryStatement": stats["num_try"] += 1
        elif name == "CatchClause": stats["num_catch"] += 1
        elif name == "ReturnStatement": stats["num_return"] += 1
        elif name == "ThrowStatement": stats["num_throw"] += 1
        elif name == "Assignment": stats["num_assignment"] += 1
        elif name == "BinaryOperation": stats["num_binary_op"] += 1

        # finally không phải lúc nào cũng là node riêng; thường nằm trong TryStatement.finally_block
        if name == "TryStatement" and getattr(node, "finally_block", None):
            stats["num_finally"] += 1

        # method calls / API calls
        if name == "MethodInvocation":
            stats["num_method_invocations"] += 1
            member = getattr(node, "member", "") or ""
            qualifier = getattr(node, "qualifier", "") or ""
            if member:
                meta["api_calls"].add(str(member))
            if qualifier:
                meta["type_names"].add(str(qualifier).split(".")[-1])

            m = str(member)
            q = str(qualifier)
            all_call_text = f"{q}.{m}"
            if m in FILE_METHOD_HINTS or any(h in all_call_text for h in FILE_IO_HINTS):
                stats["file_io_count"] += 1
            if m in STREAM_METHOD_HINTS or any(h in all_call_text for h in STREAM_IO_HINTS):
                stats["stream_io_count"] += 1
            if m in STRING_METHOD_HINTS or "String" in all_call_text or "StringBuilder" in all_call_text:
                stats["string_processing_count"] += 1
            if m in HASH_METHOD_HINTS or any(h in all_call_text for h in HASH_DIGEST_HINTS):
                stats["hash_digest_count"] += 1
            if m in COLLECTION_METHOD_HINTS or any(h in all_call_text for h in COLLECTION_HINTS):
                stats["collection_count"] += 1

        # type/class usage
        tname = get_type_name_from_node(node)
        if tname:
            meta["type_names"].add(tname)

        # variables
        if name == "MemberReference":
            member = getattr(node, "member", "") or ""
            if member:
                meta["variables"].add(str(member))
        elif name in {"VariableDeclarator", "FormalParameter"}:
            v = safe_get_name(node)
            if v:
                meta["variables"].add(v)

        # literals
        if name == "Literal":
            val = str(getattr(node, "value", ""))
            if len(val) >= 2 and (val[0] == '"' or val[0] == "'"):
                meta["string_literals"].add(val.strip('"\''))
            elif re.fullmatch(r"[-+]?\d+(\.\d+)?[a-zA-Z]*", val):
                meta["numeric_literals"].add(val)

    # Max AST depth
    def dfs_depth(n, depth=0):
        if not isinstance(n, javalang.ast.Node):
            return depth
        best = depth
        for ch in iter_children(n):
            best = max(best, dfs_depth(ch, depth + 1))
        return best

    stats["max_ast_depth"] = dfs_depth(root, 0)

    # Category counts based on type names / literals too
    type_names = meta["type_names"]
    lit_text = " ".join(meta["string_literals"])
    if type_names & FILE_IO_HINTS: stats["file_io_count"] += len(type_names & FILE_IO_HINTS)
    if type_names & STREAM_IO_HINTS: stats["stream_io_count"] += len(type_names & STREAM_IO_HINTS)
    if type_names & BUFFER_IO_HINTS: stats["buffer_io_count"] += len(type_names & BUFFER_IO_HINTS)
    if type_names & ZIP_IO_HINTS: stats["zip_io_count"] += len(type_names & ZIP_IO_HINTS)
    if type_names & URL_NETWORK_HINTS: stats["url_network_count"] += len(type_names & URL_NETWORK_HINTS)
    if type_names & XML_DOM_HINTS: stats["xml_dom_count"] += len(type_names & XML_DOM_HINTS)
    if type_names & COLLECTION_HINTS: stats["collection_count"] += len(type_names & COLLECTION_HINTS)
    if type_names & DATE_TIME_HINTS: stats["date_time_count"] += len(type_names & DATE_TIME_HINTS)
    if type_names & HASH_DIGEST_HINTS or any(h.lower() in lit_text.lower() for h in ["md5", "sha", "sha1", "sha-1", "sha256"]):
        stats["hash_digest_count"] += 1
    if any(x in lit_text.lower() for x in ["http://", "https://", "url"]):
        stats["url_network_count"] += 1
    if any(x in lit_text.lower() for x in [".xml", "<xml", "document"]):
        stats["xml_dom_count"] += 1

    stats["num_unique_api_calls"] = len(meta["api_calls"])
    stats["num_unique_types"] = len(meta["type_names"])
    stats["num_string_literals"] = len(meta["string_literals"])
    stats["num_numeric_literals"] = len(meta["numeric_literals"])
    stats["num_variables"] = len(meta["variables"])

    # DFG-level summary
    try:
        dfg = build_dfg_graph(code, max_nodes=max_dfg_nodes or cfg.max_dfg_nodes)
        stats["dfg_num_nodes"] = len(dfg.nodes)
        stats["dfg_num_edges"] = len(dfg.edges)
        n = max(len(dfg.nodes), 1)
        stats["dfg_density"] = len(dfg.edges) / float(n * max(n - 1, 1))
    except Exception:
        pass

    raw = np.array([stats[k] for k in STAT_KEYS], dtype=np.float32)
    # Log scale cho count feature để tránh feature lớn lấn át embedding.
    raw = np.log1p(raw)

    stat_vec = torch.tensor(raw, dtype=torch.float)
    # Chuyển set thành set string thuần để torch.save/load ổn định hơn
    meta = {k: _to_plain_set(v) for k, v in meta.items()}
    return stat_vec, meta


def build_pair_stat_from_meta(feat1: dict, feat2: dict) -> torch.Tensor:
    """Tạo vector similarity/difference cho một pair code."""
    meta1 = feat1.get("stat_meta", {})
    meta2 = feat2.get("stat_meta", {})

    stat1 = feat1["stat_vec"].float()
    stat2 = feat2["stat_vec"].float()
    diff = torch.abs(stat1 - stat2)

    values = [
        jaccard(meta1.get("api_calls", set()), meta2.get("api_calls", set())),
        jaccard(meta1.get("type_names", set()), meta2.get("type_names", set())),
        jaccard(meta1.get("string_literals", set()), meta2.get("string_literals", set())),
        jaccard(meta1.get("variables", set()), meta2.get("variables", set())),
        float(diff.mean().item()),
        float(diff.max().item()),
    ]
    return torch.tensor(values, dtype=torch.float)

print("STAT_VECTOR_DIM =", STAT_VECTOR_DIM)
print("PAIR_EXTRA_DIM  =", PAIR_EXTRA_DIM)


tokenizer = AutoTokenizer.from_pretrained(cfg.codebert_name)
codebert = AutoModel.from_pretrained(cfg.codebert_name).to(device)

if cfg.freeze_codebert:
    for p in codebert.parameters():
        p.requires_grad = False
    codebert.eval()

def encode_codebert(code: str, max_length: int = 256):
    inputs = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = codebert(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu()

def prepare_one_code_sample(code: str) -> dict:
    # 1. CodeBERT vector
    code_vec = encode_codebert(code, max_length=cfg.max_code_length)  # shape: (1, 768)

    # 2. AST graph
    ast_sample = build_ast_graph(code, max_nodes=cfg.max_ast_nodes)
    ast_graph = encode_ast_graph(ast_sample)
    ast_graph["batch"] = torch.zeros(ast_graph["node_type_ids"].shape[0], dtype=torch.long)

    # 3. DFG graph
    dfg_sample = build_dfg_graph(code, max_nodes=cfg.max_dfg_nodes)
    dfg_graph = encode_dfg_graph(dfg_sample)
    dfg_graph["batch"] = torch.zeros(dfg_graph["stmt_type_ids"].shape[0], dtype=torch.long)

    # 4. Statistical vector + meta sets for pair similarity
    stat_vec, stat_meta = extract_code_stats(code, max_dfg_nodes=cfg.max_dfg_nodes)

    return {
        "code_vec": code_vec,   # Tensor (1, 768)
        "ast_graph": ast_graph,
        "dfg_graph": dfg_graph,
        "stat_vec": stat_vec,   # Tensor (STAT_VECTOR_DIM,)
        "stat_meta": stat_meta, # dict of sets: api_calls/type_names/string_literals/variables
    }
