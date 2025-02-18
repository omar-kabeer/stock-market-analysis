#!/bin/bash

echo "🔹 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✅ Virtual environment activated!"

echo "🔹 Installing dependencies..."
pip install -r requirements.txt

echo "🔹 Setup complete! Run 'source venv/bin/activate' to activate."
