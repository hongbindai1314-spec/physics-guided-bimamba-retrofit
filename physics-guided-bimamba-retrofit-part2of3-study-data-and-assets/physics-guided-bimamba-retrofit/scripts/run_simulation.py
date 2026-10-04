#!/usr/bin/env python3
"""Root entry point for the benchmark pipeline under benchmark/."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['reproduce','verify','tests','bimamba'],default='reproduce')
    a,extra=ap.parse_known_args()
    sim=ROOT/'benchmark'
    if a.mode=='reproduce':cmd=[sys.executable,str(sim/'scripts/reproduce.py'),*extra]
    elif a.mode=='verify':cmd=[sys.executable,str(sim/'scripts/verify_results.py'),*extra]
    elif a.mode=='tests':cmd=[sys.executable,'-m','unittest','discover','-s',str(sim/'tests'),'-v',*extra]
    else:cmd=[sys.executable,str(sim/'scripts/train_optional_bimamba.py'),*extra]
    return subprocess.call(cmd,cwd=sim)

if __name__=='__main__':raise SystemExit(main())
