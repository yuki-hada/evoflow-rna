
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright (C) 2024 Atom Bioworks


import importlib
from omegaconf import DictConfig
import os
import glob

from src.ncrna.utils import import_modules

MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator



# automatically import any Python files in the models/ directory
import_modules(os.path.dirname(__file__), "src.ncrna.models", excludes=['protein_structure_prediction'])
