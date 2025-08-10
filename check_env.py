import importlib
import sys
import textwrap
import os

# List only the packages your project depends on
packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",        # scikit-learn
    "imblearn",       # imbalanced-learn
    "tensorflow",
    "mlflow",
    "tqdm",
    "joblib",
    "yaml",           # PyYAML
    "notebook",
    "ipykernel"
]

missing_or_broken = []

print("=" * 50)
print(f" Environment Diagnostic — {os.path.basename(sys.prefix)}")
print("=" * 50)

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f" {pkg} — OK")
    except ImportError as e:
        print(f"❌ {pkg} — FAILED: {e}")
        missing_or_broken.append(pkg)

print("=" * 50)
if missing_or_broken:
    print(f" Missing or broken packages: {missing_or_broken}")
    print(" Fix by running:")
    print(f"pip install {' '.join(missing_or_broken)}")
else:
    print(" All required packages are installed and working!")
print("=" * 50)