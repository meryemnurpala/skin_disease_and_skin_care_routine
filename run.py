"""
Uygulamayı çalıştırmak için basit başlatıcı script
"""

import subprocess
import sys
import os

def main():
    """Streamlit uygulamasını başlat"""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Hata: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nUygulama kapatılıyor...")
        sys.exit(0)

if __name__ == "__main__":
    main()

