from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=subprocess.run([sys.executable,str(ROOT/'src'/'pipeline.py')],cwd=ROOT)
    raise SystemExit(p.returncode)
