import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from services.config_generator import generate_vina_config
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during import: {e}")
    sys.exit(1)

print("Test script finished")
