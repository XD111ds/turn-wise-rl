import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    destinations = {"verl": args.verl.resolve(), "sglang": args.sglang.resolve()}
    required = {
        "verl": "verl/trainer/main_ppo.py",
        "sglang": "python/sglang/srt/models/qwen2_5_vl.py",
    }
    revisions = {
        "verl": "83ebd007e01de29bbe353de112d04245b4820b47",
        "sglang": "dcae1fb2cd2def2a2eb4919c21f74e91d888d161",
    }
    for name, directory in destinations.items():
        if not (directory / required[name]).is_file():
            raise SystemExit(f"Invalid {name} source directory: {directory}")
        if (directory / ".git").exists():
            revision = subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != revisions[name]:
                raise SystemExit(
                    f"{name} requires revision {revisions[name]}; got {revision}"
                )
    for name, directory in destinations.items():
        source = root / "overrides" / name
        for path in sorted(source.rglob("*.py")):
            target = directory / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(destinations["verl"]),
            "-e",
            str(destinations["sglang"] / "python"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
