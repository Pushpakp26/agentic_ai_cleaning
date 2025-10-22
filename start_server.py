#!/usr/bin/env python3
"""
AI Data Cleaner Server Startup Script

Run this script from the project root directory:
    python start_server.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("Starting AI Data Cleaner Server...")
    print("Server will be available at: http://localhost:8000")
    print("Web UI will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("\n" + "="*50)
    
    # Change to server directory and run uvicorn from there
    server_dir = Path(__file__).parent / "ai-data-cleaner" / "server"
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    try:
        subprocess.run(cmd, cwd=server_dir)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
