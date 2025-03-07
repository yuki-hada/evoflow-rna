import torch
from pathlib import Path
import gdown
import json

from src.rinalmo.data.alphabet import Alphabet
from src.rinalmo.model.model import RiNALMo
from src.rinalmo.config import model_config

DEFAULT_CACHE_DIR = "/home/abwer/evoflow-rna/src/rinalmo/pretrained/" #Path.home() / ".cache" / "rinalmo_pretrained"

with open(Path(__file__).parent / "resources" / "model2gdisk.json", "r") as f:
    MODEL_TO_GDISK_ID = json.load(f)
AVAILABLE_MODELS = MODEL_TO_GDISK_ID.keys()

def download_pretrained_model(model_name: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=MODEL_TO_GDISK_ID[model_name], output=str(local_path.resolve()))

def get_pretrained_model(model_name: str, force_download: bool = False, lm_config: str = "mega") -> None:
    # assert model_name in AVAILABLE_MODELS, f"Model '{model_name}' is not available! Available models: {AVAILABLE_MODELS}"
    pretrained_weights_path = DEFAULT_CACHE_DIR + f"{model_name}.pt"

    config = model_config(lm_config)
    model = RiNALMo(config)
    alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(pretrained_weights_path))

    return model, alphabet
