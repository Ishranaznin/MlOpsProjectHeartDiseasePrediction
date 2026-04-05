import sys, os
# Add src/ to path so model_utils resolves when pytest runs from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
