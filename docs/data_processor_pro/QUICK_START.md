# Data Processor Pro - Quick Start Guide

Get started with Data Processor Pro in 5 minutes!

---

## Installation

### Step 1: Install Dependencies

```bash
cd python/data_processor_pro
pip install -r requirements.txt
```

**Estimated time:** 2-3 minutes

### Step 2: Launch Application

```bash
python app.py
```

The application will open with a modern dashboard interface.

---

## First Steps

### 1. Load Your Data

**Method A: Use the GUI**
1. Click **"📁 Open Files"** in the sidebar
2. Select your data files (CSV, Excel, Parquet, etc.)
3. Files will be loaded and displayed

**Method B: Drag and Drop**
1. Drag files from your file explorer
2. Drop them onto the application window
3. Automatic format detection and loading

**Supported Formats:**
- CSV/TSV files (`.csv`, `.txt`, `.tsv`)
- Excel files (`.xlsx`, `.xls`)
- Parquet files (`.parquet`)
- JSON files (`.json`)
- And 15+ more formats!

---

### 2. Process Your Signals

#### Basic Filtering

1. Go to the **"Processing"** tab
2. Select your signal columns
3. Choose a filter type:
   - **Moving Average** - Smooth noisy data
   - **Butterworth** - Remove high/low frequencies
   - **Median Filter** - Remove outliers
   - **Kalman Filter** - Optimal filtering for Gaussian noise

4. Adjust filter parameters
5. Click **"Apply Filter"**

#### Example: Remove Noise from Temperature Data

```python
# Programmatic approach
from data_processor_pro.core import DataEngine, SignalProcessor

# Load data
engine = DataEngine()
data = engine.load_file("temperature_data.csv")

# Process
processor = SignalProcessor(use_numba=True)
temperature = engine.to_numpy("temperature")

# Apply Kalman filter (best for sensor noise)
filtered = processor.kalman_filter(
    temperature,
    process_variance=1e-5,
    measurement_variance=1e-2
)

# Save
import numpy as np
engine.save_file("filtered_temperature.parquet",
                 engine.get_data().with_columns({"filtered": filtered}))
```

---

### 3. Visualize Your Data

#### Interactive Plots

1. Go to the **"Visualization"** tab
2. Select signals to plot
3. Choose plot type:
   - Line plot (default)
   - Scatter plot
   - 3D visualization
   - Heatmap
   - Spectrogram

4. Customize:
   - Title and labels
   - Colors and themes
   - Grid and legend

5. Interact:
   - **Zoom:** Click and drag
   - **Pan:** Hold shift and drag
   - **Hover:** See exact values
   - **Export:** Click export button

#### Example: Compare Original vs Filtered

```python
from data_processor_pro.ui import InteractiveVisualizer

viz = InteractiveVisualizer(template="plotly_dark")

# Create comparison plot
fig = viz.line_plot(
    x=range(len(temperature)),
    y_data={
        "Original": temperature,
        "Kalman Filtered": filtered
    },
    title="Temperature: Original vs Filtered",
    x_label="Sample",
    y_label="Temperature (°C)"
)

# Export
viz.export_html("temperature_comparison.html")
viz.export_image("temperature_comparison.png", width=1920, height=1080)
```

---

### 4. Analyze Your Data

#### Statistical Analysis

1. Go to the **"Analytics"** tab
2. Select analysis type:
   - **Descriptive Statistics** - Mean, std, quartiles, skewness
   - **Normality Tests** - Check if data is normally distributed
   - **Correlation Analysis** - Find relationships between signals
   - **Time Series Decomposition** - Extract trend and seasonality

3. View results in interactive tables and plots

#### Example: Detect Anomalies

```python
from data_processor_pro.analytics import AnomalyDetector

detector = AnomalyDetector()

# Method 1: Statistical (fast)
anomalies, scores = detector.modified_z_score_method(temperature, threshold=3.5)

# Method 2: Machine Learning (accurate)
anomalies, scores = detector.isolation_forest(temperature, contamination=0.1)

# Method 3: Time Series (for periodic data)
anomalies, components = detector.seasonal_decompose_anomaly(
    temperature,
    period=24,  # Daily pattern
    threshold=3.0
)

print(f"Found {anomalies.sum()} anomalies")

# Visualize
viz.line_plot(
    x=range(len(temperature)),
    y_data={
        "Temperature": temperature,
        "Anomalies": temperature * anomalies  # Only show anomaly points
    },
    title="Temperature Anomalies"
)
```

---

## Common Workflows

### Workflow 1: Clean and Export Data

**Goal:** Remove outliers and export to Parquet

```python
from data_processor_pro.core import DataEngine
from data_processor_pro.analytics import AnomalyDetector

# Load
engine = DataEngine()
data = engine.load_file("raw_data.csv")

# Detect and remove outliers
detector = AnomalyDetector()
signal = engine.to_numpy("signal")
anomalies, _ = detector.iqr_method(signal, multiplier=1.5)

# Replace anomalies with interpolated values
cleaned = detector.replace_anomalies(signal, anomalies, method='interpolate')

# Save (Parquet is 20-40x faster than CSV!)
engine.get_data().with_columns({"cleaned": cleaned})
engine.save_file("cleaned_data.parquet")
```

**Performance:** CSV → Parquet conversion is **20-40x faster** for subsequent loads!

---

### Workflow 2: Batch Processing

**Goal:** Process 100 files in parallel

```python
from pathlib import Path
from data_processor_pro.core import DataEngine, SignalProcessor

engine = DataEngine(max_workers=8)
processor = SignalProcessor(use_numba=True)

# Find all CSV files
files = list(Path("data/").glob("*.csv"))

# Load all files in parallel
combined_data = engine.load_multiple(files, combine=True)

# Process all signals
for column in combined_data.columns:
    if column != "time":
        signal = engine.to_numpy(column)
        filtered = processor.moving_average(signal, window_size=10)
        # Update column...

# Save
engine.save_file("batch_processed.parquet")
```

**Performance:** 100 files processed in **seconds** vs minutes with original!

---

### Workflow 3: Professional Report

**Goal:** Generate HTML report with interactive plots

```python
from data_processor_pro.core import DataEngine
from data_processor_pro.analytics import StatisticalAnalyzer
from data_processor_pro.ui import InteractiveVisualizer

# Load and analyze
engine = DataEngine()
data = engine.load_file("experiment_data.csv")
analyzer = StatisticalAnalyzer()

signals = ["temperature", "pressure", "flow"]
stats = {}

for signal in signals:
    signal_data = engine.to_numpy(signal)
    stats[signal] = analyzer.descriptive_stats(signal_data)

# Create visualizations
viz = InteractiveVisualizer()

# Plot 1: Time series
fig1 = viz.line_plot(
    x=engine.to_numpy("time"),
    y_data={s: engine.to_numpy(s) for s in signals},
    title="Process Variables Over Time"
)

# Plot 2: Correlation matrix
correlation_data = np.column_stack([engine.to_numpy(s) for s in signals])
corr_matrix, _ = analyzer.correlation_matrix(correlation_data)
fig2 = viz.correlation_matrix(corr_matrix, signals, title="Signal Correlations")

# Plot 3: Distribution
fig3 = viz.box_plot(
    {s: engine.to_numpy(s) for s in signals},
    title="Signal Distributions"
)

# Export all
viz.export_html("fig1_timeseries.html", fig1)
viz.export_html("fig2_correlation.html", fig2)
viz.export_html("fig3_distribution.html", fig3)

print("Report generated successfully!")
```

---

## Performance Tips

### 🚀 Maximum Speed

1. **Use Parquet format:**
   - 10-100x faster loading than CSV
   - 60% less memory usage
   - Built-in compression

2. **Enable Numba JIT:**
   ```python
   processor = SignalProcessor(use_numba=True)  # Default
   ```

3. **Use Polars:**
   ```python
   engine = DataEngine(use_polars=True)  # Default
   ```

4. **Lazy loading for huge files:**
   ```python
   engine = DataEngine(lazy_mode=True)  # Default
   ```

5. **Parallel processing:**
   ```python
   engine = DataEngine(max_workers=8)  # Use all CPU cores
   ```

### 📊 Best File Format for Each Use Case

| Use Case | Recommended Format | Why |
|----------|-------------------|-----|
| Archival | Parquet (compressed) | Smallest size, fast access |
| Sharing | CSV | Universal compatibility |
| Analysis | Parquet | Fastest loading and processing |
| Large datasets (>1GB) | Parquet | Lazy loading support |
| Database export | Arrow/Feather | Zero-copy to pandas/polars |
| Machine learning | HDF5 | Random access to subsets |

---

## Keyboard Shortcuts

Master these for **10x productivity:**

| Action | Shortcut | Description |
|--------|----------|-------------|
| Open files | `Ctrl+O` | Quick file dialog |
| Save session | `Ctrl+S` | Save all state |
| Undo | `Ctrl+Z` | Revert last action |
| Redo | `Ctrl+Y` | Restore action |
| Toggle theme | `Ctrl+T` | Dark ↔ Light |
| Toggle sidebar | `Ctrl+B` | More screen space |

---

## Next Steps

### Advanced Topics

1. **Machine Learning Preprocessing** - Scale, encode, and engineer features
2. **Time Series Analysis** - Decomposition, forecasting, patterns
3. **Custom Filters** - Write your own Numba-optimized filters
4. **Plugin Development** - Extend functionality with plugins
5. **GPU Acceleration** - 100x faster with CUDA

### Example Projects

Check out `/examples/` for:
- Signal processing examples
- Anomaly detection case studies
- Batch processing scripts
- Report generation templates

---

## Getting Help

- **Documentation:** `/docs/data_processor_pro/README.md`
- **API Reference:** `/docs/data_processor_pro/API.md`
- **Troubleshooting:** `/docs/data_processor_pro/TROUBLESHOOTING.md`
- **Examples:** `/examples/data_processor_pro/`

---

**Happy analyzing! 📊**
