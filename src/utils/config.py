from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {config_path}. "
            "Pastikan config/settings.yaml sudah dibuat."
        )

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Format YAML tidak valid pada {config_path}.")

    return data