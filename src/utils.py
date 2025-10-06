# utils.py
import os

def ensure_dirs():
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
