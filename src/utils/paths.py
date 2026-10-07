import os
from pathlib import Path


def project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return str(path)
    root = Path(os.environ.get('GPSTFDIFF_ROOT', Path(__file__).resolve().parents[2])).expanduser()
    directories = {'data': 'GPSTFDIFF_DATA_DIR', 'results': 'GPSTFDIFF_RESULTS_DIR'}
    if path.parts and path.parts[0] in directories:
        override = os.environ.get(directories[path.parts[0]])
        if override:
            base = Path(override).expanduser()
            if not base.is_absolute():
                base = root / base
            return str(base.joinpath(*path.parts[1:]))
    return str(root / path)
