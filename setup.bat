@echo off
echo 🔹 Setting up Python virtual environment...
python -m venv venv
call venv\Scripts\activate
echo ✅ Virtual environment activated!

echo 🔹 Installing dependencies...
pip install -r requirements.txt


echo 🔹 Setup complete! Run 'venv\Scripts\activate' to activate.
pause
