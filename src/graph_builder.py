from .config import *

@dataclass
class ASTNodeData:
    node_id: int
    node_type: str
    node_name: str = ""
    depth: int = 0
    child_count: int = 0
    flags: Dict[str, int] = field(default_factory=dict)

@dataclass
class ASTEdgeData:
    src: int
    dst: int
    edge_type: str

@dataclass
class ASTGraphSample:
    code: str
    nodes: List[ASTNodeData]
    edges: List[ASTEdgeData]

@dataclass
class DFGNodeData:
    node_id: int
    stmt_type: str
    defs: List[str] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    text: str = ""

@dataclass
class DFGEdgeData:
    src: int
    dst: int
    edge_type: str

@dataclass
class DFGGraphSample:
    code: str
    nodes: List[DFGNodeData]
    edges: List[DFGEdgeData]

def safe_get_name(node) -> str:
    for attr in ["name", "member", "qualifier", "value"]:
        if hasattr(node, attr):
            val = getattr(node, attr)
            if isinstance(val, str):
                return val
    return ""

def iter_children(node):
    if not isinstance(node, javalang.ast.Node):
        return
    for attr_name in node.attrs:
        value = getattr(node, attr_name)
        if isinstance(value, javalang.ast.Node):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, javalang.ast.Node):
                    yield item

def walk_tree(node):
    if not isinstance(node, javalang.ast.Node):
        return
    yield node
    for child in iter_children(node):
        yield from walk_tree(child)

def parse_java_method(code: str):
    tokens = javalang.tokenizer.tokenize(code)
    parser = javalang.parser.Parser(tokens)
    return parser.parse_member_declaration()

def get_ast_flags(node) -> Dict[str, int]:
    t = type(node).__name__
    return {
        "is_call": int(t == "MethodInvocation"),
        "is_if": int(t == "IfStatement"),
        "is_loop": int(t in {"ForStatement", "WhileStatement", "DoStatement"}),
        "is_try": int(t == "TryStatement"),
        "is_catch": int(t == "CatchClause"),
        "is_return": int(t == "ReturnStatement"),
        "is_assignment": int(t == "Assignment"),
        "is_declaration": int(t in {"LocalVariableDeclaration", "VariableDeclarator", "FormalParameter"}),
    }

def build_ast_graph(code: str, max_nodes: int = 256) -> ASTGraphSample:
    root = parse_java_method(code)
    nodes, edges = [], []

    def dfs(node, depth=0, parent_id=None):
        if len(nodes) >= max_nodes or not isinstance(node, javalang.ast.Node):
            return

        node_id = len(nodes)
        children = list(iter_children(node))

        nodes.append(ASTNodeData(
            node_id=node_id,
            node_type=type(node).__name__,
            node_name=safe_get_name(node),
            depth=depth,
            child_count=len(children),
            flags=get_ast_flags(node),
        ))

        if parent_id is not None:
            edges.append(ASTEdgeData(parent_id, node_id, "PARENT_CHILD"))

        prev_child_id = None
        for child in children:
            if len(nodes) >= max_nodes:
                break
            next_id = len(nodes)
            dfs(child, depth + 1, node_id)
            if next_id < len(nodes):
                if prev_child_id is not None:
                    edges.append(ASTEdgeData(prev_child_id, next_id, "NEXT_SIBLING"))
                prev_child_id = next_id

    dfs(root, 0, None)
    return ASTGraphSample(code=code, nodes=nodes, edges=edges)

print("=====Cell 5 done")

STATEMENT_TYPES = {
    "LocalVariableDeclaration",
    "StatementExpression",
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "DoStatement",
    "TryStatement",
    "ReturnStatement",
}

def is_statement_node(node) -> bool:
    return type(node).__name__ in STATEMENT_TYPES

def extract_statements(root):
    return [node for node in walk_tree(root) if is_statement_node(node)]

def extract_identifiers(node) -> List[str]:
    vars_ = []
    for n in walk_tree(node):
        t = type(n).__name__
        if t == "MemberReference":
            if hasattr(n, "member") and isinstance(n.member, str):
                vars_.append(n.member)
        elif t == "FormalParameter":
            if hasattr(n, "name") and isinstance(n.name, str):
                vars_.append(n.name)
        elif t == "VariableDeclarator":
            if hasattr(n, "name") and isinstance(n.name, str):
                vars_.append(n.name)
    return vars_

def dedup_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def extract_def_use_from_statement(stmt):
    stmt_type = type(stmt).__name__
    defs, uses, unknown = [], [], []

    if stmt_type == "LocalVariableDeclaration":
        declarators = getattr(stmt, "declarators", []) or []
        for d in declarators:
            if hasattr(d, "name") and d.name:
                defs.append(d.name)
            if hasattr(d, "initializer") and d.initializer is not None:
                uses.extend(extract_identifiers(d.initializer))

    elif stmt_type == "StatementExpression":
        expr = getattr(stmt, "expression", None)
        if expr is not None:
            expr_type = type(expr).__name__
            if expr_type == "Assignment":
                left = getattr(expr, "expressionl", None)
                right = getattr(expr, "value", None)
                if left is not None:
                    defs.extend(extract_identifiers(left))
                if right is not None:
                    uses.extend(extract_identifiers(right))
            else:
                uses.extend(extract_identifiers(expr))

    elif stmt_type == "IfStatement":
        cond = getattr(stmt, "condition", None)
        if cond is not None:
            uses.extend(extract_identifiers(cond))

    elif stmt_type in {"ForStatement", "WhileStatement", "DoStatement", "TryStatement"}:
        uses.extend(extract_identifiers(stmt))

    elif stmt_type == "ReturnStatement":
        expr = getattr(stmt, "expression", None)
        if expr is not None:
            uses.extend(extract_identifiers(expr))
    else:
        uses.extend(extract_identifiers(stmt))

    return (
        dedup_keep_order(defs),
        dedup_keep_order(uses),
        dedup_keep_order(unknown),
    )

def build_dfg_graph(code: str, max_nodes: int = 128) -> DFGGraphSample:
    root = parse_java_method(code)
    statements = extract_statements(root)[:max_nodes]

    nodes, edges = [], []
    stmt_infos = []

    for i, stmt in enumerate(statements):
        defs, uses, unknown = extract_def_use_from_statement(stmt)
        nodes.append(DFGNodeData(
            node_id=i,
            stmt_type=type(stmt).__name__,
            defs=defs,
            uses=uses,
            unknown=unknown,
            text=type(stmt).__name__,
        ))
        stmt_infos.append({"defs": defs, "uses": uses, "unknown": unknown})

    # sequential reaching-def baseline
    for j, info in enumerate(stmt_infos):
        for used_var in info["uses"]:
            for i in range(j - 1, -1, -1):
                if used_var in stmt_infos[i]["defs"]:
                    edges.append(DFGEdgeData(i, j, "DATA_DEP"))
                    break

    return DFGGraphSample(code=code, nodes=nodes, edges=edges)

class SimpleVocab:
    def __init__(self):
        self.token2id = {"<UNK>": 0}
        self.id2token = ["<UNK>"]

    def add(self, token: str):
        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)

    def encode(self, token: str) -> int:
        return self.token2id.get(token, 0)

    def __len__(self):
        return len(self.id2token)

ast_type_vocab = SimpleVocab()
dfg_stmt_vocab = SimpleVocab()
var_name_vocab = SimpleVocab()

def build_vocabs(ast_samples, dfg_samples):
    for gs in ast_samples:
        for n in gs.nodes:
            ast_type_vocab.add(n.node_type)
    for gs in dfg_samples:
        for n in gs.nodes:
            dfg_stmt_vocab.add(n.stmt_type)
            for v in n.defs + n.uses + n.unknown:
                var_name_vocab.add(v)

def encode_ast_graph(sample: ASTGraphSample):
    edge_type_map = {"PARENT_CHILD": 0, "NEXT_SIBLING": 1}
    node_type_ids, numeric_feats, edge_index, edge_type_ids = [], [], [], []

    for node in sample.nodes:
        node_type_ids.append(ast_type_vocab.encode(node.node_type))
        numeric_feats.append([
            node.depth,
            node.child_count,
            node.flags["is_call"],
            node.flags["is_if"],
            node.flags["is_loop"],
            node.flags["is_try"],
            node.flags["is_catch"],
            node.flags["is_return"],
            node.flags["is_assignment"],
            node.flags["is_declaration"],
        ])

    for e in sample.edges:
        edge_index.append([e.src, e.dst])
        edge_type_ids.append(edge_type_map[e.edge_type])

    return {
        "node_type_ids": torch.tensor(node_type_ids, dtype=torch.long),
        "numeric_feats": torch.tensor(numeric_feats, dtype=torch.float),
        "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long),
        "edge_type_ids": torch.tensor(edge_type_ids, dtype=torch.long),
    }

def mean_var_id(vars_: List[str]) -> int:
    if not vars_:
        return 0
    ids = [var_name_vocab.encode(v) for v in vars_]
    return int(sum(ids) / len(ids))

def encode_dfg_graph(sample: DFGGraphSample):
    edge_type_map = {"DATA_DEP": 0}
    stmt_type_ids, numeric_feats, edge_index, edge_type_ids = [], [], [], []

    for node in sample.nodes:
        stmt_type_ids.append(dfg_stmt_vocab.encode(node.stmt_type))
        numeric_feats.append([
            len(node.defs),
            len(node.uses),
            len(node.unknown),
            mean_var_id(node.defs),
            mean_var_id(node.uses),
        ])

    for e in sample.edges:
        edge_index.append([e.src, e.dst])
        edge_type_ids.append(edge_type_map[e.edge_type])

    return {
        "stmt_type_ids": torch.tensor(stmt_type_ids, dtype=torch.long),
        "numeric_feats": torch.tensor(numeric_feats, dtype=torch.float),
        "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long),
        "edge_type_ids": torch.tensor(edge_type_ids, dtype=torch.long),
    }

def add_single_batch(graph_dict, key):
    n = graph_dict[key].shape[0]
    graph_dict["batch"] = torch.zeros(n, dtype=torch.long)
    return graph_dict
