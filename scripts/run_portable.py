"""App-local Windows Python entry point; also works when installed read-only."""
from pathlib import Path
import os
import sys

def main():
    base = Path(__file__).resolve().parent
    runtime = base / 'splat-backend' / '_internal' / 'cloud-runtime'
    if not (runtime / 'backend' / 'app.py').is_file():
        raise SystemExit('Incomplete application: missing cloud-runtime/backend/app.py')
    sys.path.insert(0, str(runtime))
    os.environ.setdefault('PYTHONUTF8', '1')
    from backend.app import main as serve
    serve()

if __name__ == '__main__':
    main()
