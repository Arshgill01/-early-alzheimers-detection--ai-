# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

AI model for early detection of Alzheimer's disease using cognitive tests, imaging, or other biomarkers.

## Project Structure

- `src/` - Python source code for models and utilities
- `notebooks/` - Jupyter notebooks for exploration and analysis
- `data/` - Dataset storage (gitignored - not committed)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt
```

## Development Notes

- Data files should be placed in `data/` directory (gitignored)
- Environment variables should be stored in `.env` (gitignored)
- Jupyter notebook checkpoints are gitignored
