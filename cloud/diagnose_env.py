
import sys
import os

print("="*60)
print(" DIAGNOSTIC ENV CHECK")
print("="*60)

def check_import(name):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
        return module
    except ImportError as e:
        print(f"[FAIL] {name}: {e}")
        return None
    except Exception as e:
        print(f"[FAIL] {name} (Error): {e}")
        return None

# 1. Core
import torch
print(f"[OK] torch: {torch.__version__}")
print(f"     CUDA: {torch.version.cuda}")
print(f"     GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"     Device: {torch.cuda.get_device_name(0)}")

try:
    import numpy
    print(f"[OK] numpy: {numpy.__version__}")
except ImportError:
    print("[FAIL] numpy")

# 2. Vision
check_import("cv2")
check_import("PIL")

# 3. Pydantic (Critical)
pydantic = check_import("pydantic")
if pydantic:
    try:
        from pydantic import field_validator
        print("[OK] pydantic.field_validator imported (V2 confirmed)")
    except ImportError:
        print("[FAIL] pydantic.field_validator NOT found (Likely V1 installed)")
        # Show what's actually installed for debugging
        import subprocess
        print("--- pip show pydantic ---")
        subprocess.run(["pip", "show", "pydantic"], check=False)
        print("--- pydantic file location ---")
        print(f"     File: {pydantic.__file__}")
        print("--- Tip: run 'pip uninstall -y pydantic pydantic-core && pip install pydantic>=2.5.0' ---")

# 4. Transformers & Accelerate
check_import("transformers")
check_import("accelerate")

# 5. RF-DETR & Roboflow
experiment = os.environ.get("EXPERIMENT_NAME", "all")

if experiment in ["rfdetr", "all"]:
    print(f"\n[INFO] Checking RF-DETR dependencies (Experiment: {experiment})...")
    check_import("roboflow")
    check_import("rfdetr")
    # Try to import internal stuff
    try:
        from rfdetr import RFDETRBase
        print("[OK] rfdetr.RFDETRBase imported")
    except ImportError as e:
        print(f"[FAIL] Check RF-DETR internals: {e}")
    except Exception as e:
        print(f"[FAIL] RF-DETR init error: {e}")
else:
    print(f"\n[INFO] Skipping RF-DETR checks (Experiment: {experiment})")

print("="*60)
print(" ENV CHECK COMPLETE")
print("="*60)
