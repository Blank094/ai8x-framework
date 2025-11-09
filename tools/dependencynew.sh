#!/bin/bash

# Clone the ai8x-training repository
echo "Cloning ai8x-training repository..."
git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git

# Clone the ai8x-synthesis repository
echo "Cloning ai8x-synthesis repository..."
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git

# Navigate to the ai8x-training directory
cd ai8x-training || { echo "Failed to enter ai8x-training directory"; exit 1; }

# Create and activate a virtual environment for ai8x-training
echo "Setting up virtual environment for ai8x-training..."
python -m venv .venv --prompt ai8x-training
echo "*" > .venv/.gitignore
source .venv/Scripts/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Upgrade pip, wheel, and setuptools
echo "Upgrading pip, wheel, and setuptools..."
python.exe -m pip install -U pip wheel setuptools

# Install requirements for ai8x-training
echo "Installing requirements for ai8x-training..."
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 || { echo "Failed to install requirements for ai8x-training"; exit 1; }

# Deactivate the virtual environment
deactivate

# Navigate to the ai8x-synthesis directory
cd ../ai8x-synthesis || { echo "Failed to enter ai8x-synthesis directory"; exit 1; }

# Create and activate a virtual environment for ai8x-synthesis
echo "Setting up virtual environment for ai8x-synthesis..."
python -m venv .venv --prompt ai8x-synthesis
echo "*" > .venv/.gitignore
source .venv/Scripts/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Upgrade pip, wheel, and setuptools
echo "Upgrading pip, wheel, and setuptools..."
python.exe -m pip install -U pip wheel setuptools

# Install requirements for ai8x-synthesis
echo "Installing requirements for ai8x-synthesis..."
pip install -r requirements.txt || { echo "Failed to install requirements for ai8x-synthesis"; exit 1; }

# Deactivate the virtual environment
deactivate

echo "Setup completed successfully!"