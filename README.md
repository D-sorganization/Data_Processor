# Data Processor

A comprehensive data processing application for CSV files, format conversion, and folder processing.

## Quick Start
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Install pre-commit hooks (optional)
bash scripts/setup_precommit.sh

# 3) Launch the application
cd python/data_processor
python launch_integrated.py
```

## Features

- **CSV Processing**: Load, process, and analyze CSV files with advanced filtering and signal processing
- **Format Converter**: Convert between 15+ file formats (CSV, Excel, Parquet, JSON, HDF5, etc.)
- **Parquet Analyzer**: Analyze Parquet file metadata and structure
- **Folder Tool**: Comprehensive folder processing (combine, flatten, deduplicate, analyze)
- **DAT File Import**: Import and process DAT files
- **Plotting & Analysis**: Interactive plotting with matplotlib integration
- **Batch Processing**: Process multiple files with progress tracking

## Daily Safety
- Commit every ~30 minutes (`wip:` if tests fail).
- End-of-day snapshot: `bash scripts/snapshot.sh`.
- Big AI refactor? Create `backup/before-ai-<desc>` branch first.

## Reproducibility
- Python env pinned via `requirements.txt` or `python/requirements.txt`.
