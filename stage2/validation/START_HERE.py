"""Run in PyMOL from any checkout to generate inspection sessions."""
import os
import inspect
from pathlib import Path
os.environ["ACCESSFOLD_GUI"] = "1"
task_script = Path(inspect.currentframe().f_code.co_filename).resolve().with_name("pymol_check.py")
exec(compile(task_script.read_text(), str(task_script), "exec"),
     {"__file__": str(task_script), "__name__": "__main__"})
