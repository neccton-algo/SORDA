#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import shutil

def run_assimilation(
    date: str,
    workdir: str = "current_day",
    exe: str = "assim-gfortran-double-mpi",  # just the name, not path
    ranks: int = 4,
):
    wd = Path(workdir).resolve()
    exe_path = (wd / exe).resolve()
    namelist = wd / f"assim.{date}"

    if not wd.exists():
        raise FileNotFoundError(f"Working directory not found: {wd}")
    if not namelist.exists():
        raise FileNotFoundError(f"Namelist not found: {namelist}")
    if not exe_path.exists():
        raise FileNotFoundError(f"Executable not found: {exe_path}")

    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun and ranks >= 1:
        cmd = [mpirun, "-np", str(ranks), str(exe_path), f"assim.{date}", "001"]
    else:
        cmd = [str(exe_path), f"assim.{date}", "001"]

    print(f"[job] Running: {' '.join(cmd)} in {wd}")
    subprocess.run(cmd, cwd=wd, check=True)
    print("[job] Done ✅")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/python/job.py YYYYMMDD [ranks]")
        sys.exit(1)
    date = sys.argv[1]
    ranks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_assimilation(date, ranks=ranks)
