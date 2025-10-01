# CV-2-03: Adaptive Canny Edge Detection

Interactive program for edge detection on images with real-time adaptive threshold adjustment for the Canny algorithm.

## Description

The program provides a graphical interface with trackbars for dynamic adjustment of Canny algorithm parameters. Supports working with static images, files, and live camera video streams.

### Requirements

- Python 3.12+
- Dependencies are declared in `pyproject.toml`

## Setup

Install uv(if needed) and sync dependencies:

```
git clone https://github.com/tywin812/CV-2-03.git
cd CV-2-03

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create .venv and install project deps
uv sync
source .venv/bin/activate

```

## Usage

Flags:
- '-c' — Use camera instead of image 
- '-i, --image' — Path to image file

```
# Use built-in test image (coins)
python src/adaptive_canny.py

# Use camera (video stream)
python src/adaptive_canny.py -c

# Load image from file
python src/adaptive_canny.py -i data/test_image.jpg
python src/adaptive_canny.py --image data/test_image.jpg
