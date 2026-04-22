import shutil
from pathlib import Path


def setup_results_dir() -> Path:
    """
    Automatically create or reset the results directory.
    If it exists, delete and recreate it.
    """
    results_dir = Path(__file__).parent / "results"
    
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"已清除现有结果目录：{results_dir}")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"已创建结果目录：{results_dir}")
    
    return results_dir
