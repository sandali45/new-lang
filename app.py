# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from lark import Lark, Token, Tree, UnexpectedInput
from graphviz import Digraph
import os

app = FastAPI(title="TinyLang Analyzer")

# ---------------------------
# Serve frontend folder
# ---------------------------
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def root():
    index_file = os.path.join(frontend_path, "index.html")
    return FileResponse(index_file)

# ---------------------------
# Enable CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Grammar (TinyLang)
# ---------------------------
GRAMMAR = r"""
start: stmt*

?stmt: declaration
    | assign ";"
    | print_stmt ";"
    | if_stmt
    | while_stmt

declaration: "int" NAME ";"    -> declaration
assign: NAME "=" expr          -> assign
print_stmt: "print" "(" expr ")"  -> print_stmt

if_stmt: "if" "(" expr ")" block ("else" block)?  -> if_stmt
while_stmt: "while" "(" expr ")" block            -> while_stmt
block:  "{" stmt* "}"

?expr: expr "+" term           -> add
    | expr "-" term           -> sub
    | expr ">" term           -> gt
    | expr "<" term           -> lt
    | expr ">=" term          -> ge
    | expr "<=" term          -> le
    | expr "==" term          -> eq
    | expr "!=" term          -> ne
    | term

?term: term "*" factor         -> mul
    | term "/" factor          -> div
    | term "%" factor          -> mod
    | factor

?factor: NUMBER                -> number
      | NAME                  -> var
      | "(" expr ")"

%import common.CNAME -> NAME
%import common.NUMBER
%import common.WS
%ignore WS
COMMENT: /\/\/[^\n]*/
%ignore COMMENT
"""

parser = Lark(GRAMMAR, parser="lalr", propagate_positions=True)

# ---------------------------
# Token map (human-readable)
# ---------------------------
TOKEN_MAP = {
    "NAME": "Identifier",
    "NUMBER": "Number",
    "INT": "Keyword",
    "PRINT": "Keyword",
    "IF": "Keyword",
    "ELSE": "Keyword",
    "WHILE": "Keyword",
    "PLUS": "Operator",
    "MINUS": "Operator",
    "STAR": "Operator",
    "SLASH": "Operator",
    "PERCENT": "Operator",
    "EQ": "Operator",
    "GT": "Operator",
    "LT": "Operator",
    "GE": "Operator",
    "LE": "Operator",
    "NE": "Operator",
    "LPAR": "Delimiter",
    "RPAR": "Delimiter",
    "LBRACE": "Delimiter",
    "RBRACE": "Delimiter",
    "SEMI": "Delimiter",
    "COMMENT": "Comment",
}

# ---------------------------
# Request/Response Models
# ---------------------------
class AnalyzeRequest(BaseModel):
    source: str

class TokenOut(BaseModel):
    type: str
    value: str
    line: Optional[int] = None
    column: Optional[int] = None

class ErrorOut(BaseModel):
    kind: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None

class AnalyzeResponse(BaseModel):
    tokens: List[TokenOut]
    errors: List[ErrorOut]
    tree: Dict[str, Any]
    svg: str

# ---------------------------
# Helper functions
# ---------------------------
def tokens_only(text: str) -> List[Token]:
    return list(parser.lex(text))

def tree_to_json(t: Tree) -> Dict[str, Any]:
    def walk(n, idx=0):
        if isinstance(n, Tree):
            node = {"id": f"n{idx}", "label": n.data, "children": []}
            next_idx = idx + 1
            for c in n.children:
                child, next_idx = walk(c, next_idx)
                node["children"].append(child)
            return node, next_idx
        else:
            node = {"id": f"n{idx}", "label": f"{n.type}:{str(n)}", "children": []}
            return node, idx + 1
    root, _ = walk(t, 0)
    return root

def tree_to_svg(t: Tree) -> str:
    dot = Digraph("ParseTree", format="svg")
    counter = [0]
    def add(n):
        nid = f"n{counter[0]}"
        counter[0] += 1
        if isinstance(n, Tree):
            dot.node(nid, n.data, shape="ellipse")
            for ch in n.children:
                cid = add(ch)
                dot.edge(nid, cid)
        else:
            dot.node(nid, f"{n.type}\n{str(n)}", shape="box")
        return nid
    add(t)
    svg_bytes = dot.pipe(format="svg")
    return svg_bytes.decode("utf-8")

def _format_syntax_error(e: UnexpectedInput, text: str) -> Dict:
    msg = f"Syntax error: unexpected token at line {getattr(e,'line','?')}, column {getattr(e,'column','?')}."
    lines = text.splitlines()
    if getattr(e, "line", None) and e.line-1 < len(lines):
        msg += f" Line content: '{lines[e.line-1].strip()}'"
    return {
        "kind": "syntax",
        "message": msg,
        "line": getattr(e, "line", None),
        "column": getattr(e, "column", None)
    }

# ---------------------------
# Semantic checker
# ---------------------------
def check_semantics(tree: Tree) -> List[Dict]:
    errors = []
    declared_vars = set()
    
    def visit(node):
        if isinstance(node, Tree):
            if node.data == "declaration":
                var_token = node.children[0]
                declared_vars.add(var_token.value)
            elif node.data == "assign":
                var_token = node.children[0]
                if var_token.value not in declared_vars:
                    errors.append({
                        "kind": "semantic",
                        "message": f"Variable '{var_token.value}' assigned before declaration.",
                        "line": var_token.line,
                        "column": var_token.column
                    })
            elif node.data == "var":
                var_token = node.children[0] if isinstance(node.children[0], Token) else node.children[0]
                if isinstance(var_token, Token) and var_token.value not in declared_vars:
                    errors.append({
                        "kind": "semantic",
                        "message": f"Variable '{var_token.value}' used before declaration.",
                        "line": var_token.line,
                        "column": var_token.column
                    })
            for child in node.children:
                visit(child)
    
    visit(tree)
    return errors

# ---------------------------
# Analyze endpoint
# ---------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.source or ""

    # Blank input handling
    if text.strip() == "":
        return AnalyzeResponse(
            tokens=[],
            errors=[{"kind": "input", "message": "Input is blank."}],
            tree={},
            svg=""
        )

    # Tokens
    tokens_out = [
        TokenOut(
            type=TOKEN_MAP.get(t.type, t.type),
            value=str(t),
            line=getattr(t, "line", None),
            column=getattr(t, "column", None)
        )
        for t in tokens_only(text)
    ]
    
    errors: List[Dict] = []
    parsed_tree: Optional[Tree] = None

    # Parse
    try:
        parsed_tree = parser.parse(text)
    except UnexpectedInput as e:
        errors.append(_format_syntax_error(e, text))
        return AnalyzeResponse(tokens=tokens_out, errors=errors, tree={}, svg="")

    # Semantic errors
    semantic_errors = check_semantics(parsed_tree)
    errors.extend(semantic_errors)

    tree_json = tree_to_json(parsed_tree)
    svg = tree_to_svg(parsed_tree)
    
    return AnalyzeResponse(tokens=tokens_out, errors=errors, tree=tree_json, svg=svg)
