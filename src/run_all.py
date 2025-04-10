#!/usr/bin/env python3
"""
Run all python scripts excluding the data loader which must be done first.
"""
import subprocess
import sys
import os

def run_script(script_name):
    """
    Runs a Python script with the same interpreter that is running this script.
    If the script fails, an exception is raised.
    """
    # Get the current directory (assumes all scripts are in the same folder as run_all.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    print(f"\n--- Running {script_name} ---\n")
    # Run the script and check for errors.
    subprocess.run([sys.executable, script_path], check=True)

def main():
    # List of scripts in the desired execution order.
    # Adjust names/paths if your file organization is different.
    scripts = [
        "train_CNN.py",
        "evaluate_CNN.py",
        "train_XGBoost.py",
        "train_VIT.py",
        "evaluate_VIT.py",
        "CNN_Modified_Input.py",
        "evaluate_CNN_Modified_Input.py"
    ]
    
    # Execute each script in order.
    for script in scripts:
        try:
            run_script(script)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
            sys.exit(1)
    print("\nAll scripts executed successfully!")

if __name__ == "__main__":
    main()
