#!/usr/bin/env python3
"""
AI Data Cleaner Server Startup Script

Run this script from the project root directory:
    python run_server.py

Or run directly from the server directory:
    cd server && python main.py
"""

import sys
import os
from pathlib import Path

# Add the server directory to Python path
server_dir = Path(__file__).parent / "ai-data-cleaner" / "server"
sys.path.insert(0, str(server_dir))

# Change to server directory to resolve relative imports
os.chdir(server_dir)

# Import and run the server
from main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting AI Data Cleaner Server...")
    print("📊 Server will be available at: http://localhost:8000")
    print("🌐 Web UI will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("\n" + "="*50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
