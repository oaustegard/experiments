import ast, sys
for path in sys.argv[1:]:
    try:
        src = open(path).read()
        tree = ast.parse(src)
    except Exception as e:
        print("PARSE-ERR", path, e); continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if body and all(isinstance(s, ast.Pass) for s in body) and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                print(f"{path}:{node.lineno} {node.name}")
