from pathlib import Path
import yaml

with open(Path(__file__).parent / "default_conf.yml", "r") as f:
    default_conf = yaml.safe_load(f)

