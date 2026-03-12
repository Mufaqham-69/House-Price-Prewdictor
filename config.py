from pathlib import Path
ROOT_DIR = Path(__file__).parent 
DATA_DIR = ROOT_DIR / "data" 
MODEL_DIR = ROOT_DIR / "models" 
RAW_DATA = DATA_DIR / "raw/train.csv" 
PROCESSED_DATA = DATA_DIR / "processed/features.csv" 
MODEL_PATH = MODEL_DIR / "best_model.pkl"
TARGET_COL = "SalePrice"