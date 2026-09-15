import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
EXAMPLES_BIN_DIR = ROOT / "install" / "bin" / "examples"


def main() -> int:
    subprocess.run((EXAMPLES_BIN_DIR / "in-mem-poc",), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
