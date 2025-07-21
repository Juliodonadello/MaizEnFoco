import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    current_dir = os.path.dirname(__file__)
    app_path = os.path.join(current_dir, "streamlit_app.py")
    
    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    print("Lanzando aplicación Streamlit...")
    print(f"Comando: {' '.join(cmd)}")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
