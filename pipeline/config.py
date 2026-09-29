import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(path=None):
    p = pathlib.Path(path) if path else ROOT / "channel.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
