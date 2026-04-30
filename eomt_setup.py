import yaml
import sys
import os
import torch
import importlib
import warnings
from lightning import seed_everything

def setup_environment(eomt_path="eomt"):
    """Add eomt to sys.path and set up seeds."""
    abs_eomt_path = os.path.abspath(eomt_path)
    if abs_eomt_path not in sys.path:
        sys.path.insert(0, abs_eomt_path)
    seed_everything(0, verbose=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    return device

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_data(config, data_path="../datasets/cityscapes"):
    """Initialize and setup the data module."""
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    
    # Reload module if already imported to catch file changes
    if data_module_name in sys.modules:
        importlib.reload(sys.modules[data_module_name])
    
    data_module_cls = getattr(importlib.import_module(data_module_name), class_name)
    data_module_kwargs = config["data"].get("init_args", {}).copy()
    
    data = data_module_cls(
        path=data_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_module_kwargs
    )
    data.setup()
    return data

def load_model(config, data, device, weights_path=None):
    """Initialize the model and load weights if provided."""
    # 1. Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    
    if encoder_module_name in sys.modules:
        importlib.reload(sys.modules[encoder_module_name])
        
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=data.img_size, **encoder_cfg.get("init_args", {}))

    # 2. Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    
    if network_module_name in sys.modules:
        importlib.reload(sys.modules[network_module_name])
        
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=data.num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # 3. Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    
    if lit_module_name in sys.modules:
        importlib.reload(sys.modules[lit_module_name])
        
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = config["model"].get("init_args", {}).copy()
    if "network" in model_kwargs:
        del model_kwargs["network"]
    
    # Handle stuff_classes if present in data config (linked in LightningCLI)
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = lit_cls(
        img_size=data.img_size,
        num_classes=data.num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    # 4. Load weights
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    elif weights_path:
        print(f"Warning: Weights file not found at {weights_path}")

    return model
