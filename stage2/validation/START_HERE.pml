python
import os
from pathlib import Path
os.environ["ACCESSFOLD_GUI"] = "1"
task_script = Path.cwd() / "pymol_check.py"
exec(compile(task_script.read_text(), str(task_script), "exec"), {"__file__": str(task_script), "__name__": "__main__"})
python end
