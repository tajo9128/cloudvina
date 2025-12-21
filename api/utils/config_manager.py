import json
import os
from datetime import datetime

CONFIG_FILE = "server_config.json"

DEFAULT_CONFIG = {
    "phases": {
        "docking": {"enabled": True, "limit_per_user": 10, "maintenance_mode": False},
        "md_simulation": {"enabled": True, "limit_per_user": 2, "maintenance_mode": False},
        "trajectory_analysis": {"enabled": True, "maintenance_mode": False},
        "binding_energy": {"enabled": True, "maintenance_mode": False},
        "lead_ranking": {"enabled": True, "maintenance_mode": False},
        "admet_prediction": {"enabled": True, "maintenance_mode": False},
        "target_prediction": {"enabled": True, "maintenance_mode": False},
        "benchmarking": {"enabled": True, "maintenance_mode": False},
        "reporting": {"enabled": True, "maintenance_mode": False}
    },
    "pricing": {
        "currency": "USD",
        "free_tier": {
            "docking_credits": 100,
            "md_hours": 10,
            "storage_gb": 1
        },
        "pro_tier": {
            "docking_cost_per_run": 0.10,
            "md_cost_per_hour": 2.00,
            "monthly_fee": 29.99
        }
    },
    "last_updated": datetime.utcnow().isoformat()
}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG

def save_config(config_data):
    config_data["last_updated"] = datetime.utcnow().isoformat()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)
    return config_data

def update_section(section, data):
    config = load_config()
    if section not in config:
        config[section] = {}
    
    # Deep update/merge could be better but shallow replace is safer for admin usage usually
    # For now, let's just update keys provided
    config[section].update(data)
    save_config(config)
    return config
