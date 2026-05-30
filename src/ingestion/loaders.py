from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
load_dotenv()


def load_file():
    file_path = os.getenv("file_path")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        df = pd.read_excel(path)

    return df