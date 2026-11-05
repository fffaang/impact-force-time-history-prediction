"""Compatibility entry point used by a new training run's export stage."""
from pathlib import Path
from export_workbooks import export


def prepare():
    export(Path(__file__).resolve().parents[1]/'output/workbooks')


if __name__=='__main__':prepare()
