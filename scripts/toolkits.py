from pathlib import Path
from datetime import datetime

def mk_fpath(dirs: str, fname: str) -> Path:
    """Make and return absolute file path.

    Args:
        dirs: 'directories/from/project root/to/file'
        fname: 'file name'

    Returns:
        'dirs/fname'
    """
    # Get the project root
    project_root = next(p for p in Path(__file__).resolve().parents if (p / 'scripts').exists())
    file_dir = project_root / dirs
    file_dir.mkdir(parents=True, exist_ok=True)
    return file_dir / fname

def format_datetime(dt_str):
    """Format date time from:
        "Thu Sep 24 11:25:33 2026 +0800"
        to
        "2026-09-24 11:25:33"
    """
    dt_obj = datetime.strptime(dt_str, "%a %b %d %H:%M:%S %Y %z")
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
