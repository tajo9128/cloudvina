
import sys
try:
    import fastapi
    import sqlalchemy
    import pydantic
    import uvicorn
    print("Environment Verification: SUCCESS")
except ImportError as e:
    print(f"Environment Verification: FAILED ({e})")
    sys.exit(1)
