**Please open the Firmware folder for device setup for the project**

# README - Python Script (`iot_fingerprinting.py`)

## Project

**Identifying Deployment Environments and IoT Sensor Nodes Using Link Quality Fluctuations**

TU Dresden | MSc Nanoelectronic Systems | IEEE 802.15.4 | Adafruit Feather nRF52840

---

## What This Script Does

This is the standalone Python version of `iot_fingerprinting.ipynb`. It runs the exact same code - same models, same preprocessing, same experiments, same outputs - but as a single command-line script instead of an interactive notebook. Use this when you want to run the full pipeline unattended (e.g., on a server or in a batch job).

The script classifies RSSI time series from 3 IoT sensor nodes across 5 environments:

- **Scenario I** - Environment classification (5-class: bridge, forest, garden, lake, river)
- **Scenario II** - Node classification (3-class: A, B, C)

Each scenario is tested with:
- **Strategy 1** - 75/25 random train/test split (seen data)
- **Strategy 2** - Leave-one-environment-out cross-validation (unseen data)

Total: **32 experiments** (2 scenarios x 2 strategies x 2 frame sizes x 2 overlaps x 2 models)

---

## Requirements

```bash
pip install torch torchinfo numpy pandas matplotlib scikit-learn
```

For CPU-only PyTorch (smaller, ~200MB instead of ~2GB):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchinfo numpy pandas matplotlib scikit-learn
```

Python 3.8+ required.

---

## File Structure

```
project/
|---iot_fingerprinting.py       # this script
|--- data_folder/                       # folder with 15 CSV files
|   |-- nodeA_bridge_main_*.csv
|   |-- nodeA_forest_main_*.csv
|   |-- ...
|   |-- nodeC_river_main_*.csv
|--- Firmware/    # Scripts and files for setting up the devices ( Rpi and Adafruit Boards)
|--- Results/     # Contains results from the script
|--- iot_fingerprinting.ipynb     # same script like the .py file but can be used with jupyter notebook  
```

Each CSV has columns: `timestamp, node_id, rssi, lqi` (18,000 rows per file).

---

## How to Run

**Basic usage** (data in `./data_folder/`):
```bash
python iot_fingerprinting.py
```

**Runtime:** approximately 30-60 minutes on CPU, 10-15 minutes on GPU depending (Tested on NVIDIA RTX 4060Ti 16GB)

The script prints progress to stdout and saves all outputs to the current working directory.

---

## What It Does Step by Step

The script runs 5 stages in sequence:

### Stage 1 - Load Data
Reads all 15 CSV files from the data directory. Extracts the node ID (A/B/C) and environment name (bridge/forest/garden/lake/river) from each filename.

### Stage 2 - Preprocess
For each (node, environment) pair:
1. **Interpolate** NaN values (packet losses, ~0.5% of data)
2. **Differentiate:** `y[i] = x[i+1] - x[i]` - removes absolute RSSI level, focuses on signal changes
3. **Normalise** to [0, 1] using min-max scaling - removes radio sensitivity bias

This follows the project specification exactly.

### Stage 3 - Print Model Summaries
Shows the architecture and parameter count for both models:
- **CNN1D:** 3-layer 1D CNN (~27K parameters)
- **ResNet1D:** 3 residual blocks (~57K parameters)

### Stage 4 - Run 32 Experiments
For each combination of (scenario, frame_size, overlap, model):
- **Strategy 1:** Random 75/25 split, train, evaluate, store confusion matrix
- **Strategy 2:** 5-fold leave-one-environment-out, train each fold, aggregate results

### Stage 5 - Generate Outputs
Saves confusion matrix plots, per-class accuracy breakdown, results table, and bar chart.

---

## Techniques Used

### Preprocessing
| Step | Formula | Purpose |
|------|---------|---------|
| Differentiation | `y_i = x_{i+1} - x_i` | Removes absolute RSSI level and distance effects |
| Normalisation | `z_i = (y_i - y_min) / (y_max - y_min)` | Scales to [0,1], removes radio sensitivity bias |
| Framing | 500 and 1000 samples, 40% and 50% overlap | Creates fixed-size inputs for the CNN |

### Models
| Model | Architecture | Parameters |
|-------|-------------|------------|
| CNN1D | 3 conv layers (32->64->128 channels) + 2 FC layers | ~27K |
| ResNet1D | 1 stem conv + 3 residual blocks (64 channels) + 2 FC layers | ~57K |

Both use `AdaptiveAvgPool1d(1)` so they work with any frame size.

### Training
| Setting | Value |
|---------|-------|
| Optimiser | Adam (lr=1e-3) |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR (T_max=60) |
| Loss | CrossEntropyLoss |
| Epochs | 100 |
| Batch size | 64 |
| Reproducibility | All seeds fixed (torch, numpy, DataLoader) |

### Evaluation Strategies
| Strategy | Method | What it tests |
|----------|--------|---------------|
| Strategy 1 | 75/25 stratified random split | Can the model learn patterns from the same data distribution? |
| Strategy 2 | Leave-one-environment-out (5 folds) | Do learned features generalise to completely unseen environments? |

---

## Output Files

| File | Description |
|------|-------------|
| `confusion_matrices_strategy1.png` | 2x2 grid: CNN1D and ResNet1D for both scenarios, Strategy 1 |
| `confusion_matrices_strategy2.png` | 2x2 grid: same layout, Strategy 2 (aggregated across 5 folds) |
| `results_comparison.png` | Side-by-side bar chart of all 32 experiments |
| `results_summary.csv` | Machine-readable table of all results |

---

## How to Read the Results

### Console Output
Each experiment prints a one-line summary:
```
>> CNN1D | Strategy 1 (Seen) | Scenario I (Environment) | frame=500 | overlap=40% | Accuracy: 90.2%
```

### Results Table (printed + CSV)
Columns: `Model, Scenario, Strategy, Frame, Overlap, Accuracy`

Look for these patterns:
- **Scenario I, Strategy 1:** High accuracy (84-96%) = environments are distinguishable
- **Scenario I, Strategy 2:** 0% = held-out class never seen during training (expected)
- **Scenario II, Strategy 1:** High accuracy (79-99%) = node fingerprints exist in the data
- **Scenario II, Strategy 2:** Below chance (<33%) = node fingerprints don't survive cross-environment domain shift

### Confusion Matrices
- **Diagonal** = correctly classified samples
- **Off-diagonal** = misclassifications
- Blue colormap = Strategy 1, Orange = Strategy 2

### Bar Chart
- **Blue bars** = Strategy 1 (seen data)
- **Orange bars** = Strategy 2 (unseen data)
- **Dashed line** = chance level

---

## Key Scientific Findings

### Finding 1: Environment classification works within the same distribution
Scenario I Strategy 1 achieves 84-96% accuracy. The differentiated and normalised RSSI signal retains enough environment-specific variance (noise amplitude, zero-change patterns) for the CNN to discriminate.

### Finding 2: Environment fingerprints do not generalise
Scenario I Strategy 2 gives 0% accuracy. This is not a bug - when one environment is entirely held out, the model has zero training examples for that class label and cannot predict it.

### Finding 3: Node fingerprints are learnable but fragile
Scenario II Strategy 1 reaches up to 99% (ResNet1D, frame=1000), proving that manufacturing differences between nRF52840 boards create learnable signatures. But Strategy 2 drops to 18-22%, indicating these signatures are entangled with environment-specific features and do not transfer.

### Finding 4: ResNet1D consistently outperforms CNN1D
Across all configurations, ResNet1D achieves higher accuracy than CNN1D, likely due to the residual connections enabling better gradient flow during training on these relatively long sequences.

---

## Differences from the Notebook

| Aspect | Notebook | Python Script |
|--------|----------|---------------|
| Execution | Cell by cell, interactive | Single `python` command, runs end-to-end |
| Output | Inline plots in browser | PNG files saved to disk |
| Data directory | Hardcoded `./data_folder` | Configurable via command-line argument |
| Extra output | - | Also saves `results_summary.csv` |
| Model summaries | Inline print | Printed to console |
| Code logic | Identical | Identical |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Install missing package: `pip install <package>` |
| `No CSV files found` | Pass the correct data directory: `python iot_fingerprinting.py ./data_folder` |
| Script crashes mid-run | Check available RAM - 32 experiments with frame=1000 need ~2GB |
| Want faster results | Reduce `EPOCHS` from 60 to 30 at the top of the script |
| Need GPU | Install CUDA-enabled PyTorch: `pip install torch` (with CUDA) |
