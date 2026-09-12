"""Verify the upstream launch boundary without pretending to run CUDA."""
import ast
from pathlib import Path
import subprocess
import sys

def test_upstream_runpy_wrapper_preserves_script_and_arguments(tmp_path):
    source=Path(__file__).resolve().parents[1]/'backend/upstream.py'
    tree=ast.parse(source.read_text())
    expression=next(node.value for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='runner' for x in node.targets))
    wrapper=ast.literal_eval(expression.elts[2])
    script=tmp_path/'train.py';script.write_text('import sys,json;print(json.dumps(sys.argv))')
    result=subprocess.run([sys.executable,'-c',wrapper,str(tmp_path),str(script),'-s','dataset with spaces'],capture_output=True,text=True,check=True)
    import json
    assert json.loads(result.stdout)==[str(script),'-s','dataset with spaces']
