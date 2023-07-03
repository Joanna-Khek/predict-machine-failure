from pathlib import Path
import yaml

with open(Path("configs/config.yaml")) as file:
    config = yaml.safe_load(file)