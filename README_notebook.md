**Please open the Firmware folder for device setup for the project**

# README - Jupyter Notebook (`iot_fingerprinting.ipynb`)

## Project

**Identifying Deployment Environments and IoT Sensor Nodes Using Link Quality Fluctuations**

TU Dresden | MSc Nanoelectronic Systems | IEEE 802.15.4 | Adafruit Feather nRF52840

---

## What This Notebook Does

This notebook classifies IoT sensor data using deep learning. It takes RSSI (Received Signal Strength Indicator) time series collected from 3 sensor nodes deployed across 5 outdoor environments and answers two questions:

1. **Scenario I** - Can we identify *which environment* the data came from? (5-class classification)
2. **Scenario II** - Can we identify *which sensor node* transmitted the data? (3-class classification)

Each scenario is tested under two conditions:

- **Strategy 1 (Seen):** Train and test data come from the same dataset (75/25 random split)
- **Strategy 2 (Unseen):** Train on 4 environments, test on the 5th that the model has never seen

---

## Requirements

Install all dependencies with:

```bash
pip install torch torchinfo numpy pandas matplotlib scikit-learn
```

For CPU-only PyTorch (smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchinfo numpy pandas matplotlib scikit-learn
```

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
```

Each CSV file has columns: `timestamp, node_id, rssi, lqi`
- 15 files total (3 nodes x 5 environments)
- 18,000 rows each (10 packets/second x 30 minutes)

---

## How to Run

1. Place the `data_folder/` folder in the same directory as the notebook
2. Open the notebook in Jupyter
3. **Run All Cells** (Kernel -> Restart & Run All)
4. Total runtime: approximately 30-60 minutes on CPU, 10-15 minutes on GPU

---

## Cell-by-Cell Guide

| Cell | Purpose |
|------|---------|
| **Cell 1** | Imports and sets random seeds for reproducibility |
| **Cell 2** | Loads all 15 CSV files, tags each row with node and environment |
| **Cell 3** | Defines preprocessing: NaN interpolation -> differentiation -> normalisation -> framing |
| **Cell 4** | Applies preprocessing to all 15 series |
| **Cell 5** | Builds labelled datasets for Scenario I and II |
| **Cell 6** | Defines CNN1D and ResNet1D model architectures, prints summaries |
| **Cell 7** | Defines training loop (Adam + CosineAnnealingLR) and evaluation |
| **Cell 8** | **Main experiment loop** - runs all 32 experiments |
| **Cell 9** | Plots confusion matrices for Strategy 1 (Seen) |
| **Cell 10** | Plots confusion matrices for Strategy 2 (Unseen) |
| **Cell 11** | Prints per-class accuracy breakdown |
| **Cell 12** | Prints the full results summary table |
| **Cell 13** | Plots the bar chart comparing all experiments |

---

## Preprocessing Pipeline

The preprocessing follows the project specification exactly:

```
Raw RSSI -> Interpolate NaN -> Differentiate (y_i = x_{i+1} - x_i)
         -> Normalise to [0,1] (per series)
         -> Segment into overlapping frames
```

- **Differentiation** removes the effect of absolute RSSI level and transmission distance, focusing on signal *fluctuations*
- **Normalisation** removes radio sensitivity bias (each series is normalised independently using its own min/max)
- **Framing** creates fixed-length input windows for the CNN. Frame sizes of 500 and 1000 samples are tested, with 40% and 50% overlap

---

## Models

### CNN1D
A 3-layer 1D convolutional network:
- Conv1d(1->32, k=7) -> BN -> ReLU -> MaxPool
- Conv1d(32->64, k=5) -> BN -> ReLU -> MaxPool
- Conv1d(64->128, k=3) -> BN -> ReLU -> AdaptiveAvgPool
- FC(128->64) -> ReLU -> Dropout(0.3) -> FC(64->n_classes)

### ResNet1D
A residual network with 3 residual blocks:
- Stem: Conv1d(1->64, k=7) -> BN -> ReLU -> MaxPool
- 3x ResBlock(64): two Conv1d(64->64, k=3) layers with skip connection
- Head: AdaptiveAvgPool -> FC(64->32) -> ReLU -> Dropout(0.3) -> FC(32->n_classes)

### Training Configuration
- Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=60)
- Loss: CrossEntropyLoss
- Epochs: 60
- Batch size: 64
- All random seeds fixed for reproducibility

---

## How to Read the Results

### The Summary Table
The table has columns: `Model, Scenario, Strategy, Frame, Overlap, Accuracy`

- **32 rows** = 2 models x 2 scenarios x 2 strategies x 2 frame sizes x 2 overlaps

### Confusion Matrices
- **Strategy 1 (Blues):** Shows how well the model classifies when it has seen similar data during training. Diagonal = correct, off-diagonal = errors.
- **Strategy 2 (Oranges):** Aggregated across all 5 leave-one-out folds. Shows how well features generalise to unseen environments.

### Bar Chart
- **Blue bars** = Strategy 1 (Seen data)
- **Orange bars** = Strategy 2 (Unseen data)
- **Dashed line** = chance level (20% for 5-class, 33% for 3-class)

### What to Expect

| Scenario | Strategy | Typical Accuracy | Meaning |
|----------|----------|-----------------|---------|
| I (Environment) | 1 (Seen) | 84-96% | Environments are distinguishable within the same data distribution |
| I (Environment) | 2 (Unseen) | 0% | Held-out environment was never seen during training - model cannot predict it |
| II (Node) | 1 (Seen) | 79-99% | Node fingerprints are learnable within the same distribution |
| II (Node) | 2 (Unseen) | 18-22% | Node fingerprint does not survive the domain shift to unseen environments |

### Key Findings

**Scenario I Strategy 2 = 0%** is structurally guaranteed. In leave-one-environment-out, the test set contains *only* the held-out class. Since the model never trained on that class, it cannot predict it. This confirms that environment fingerprints do not generalise across locations.

**Scenario II Strategy 2 = 18-22%** (below the 33% chance level) means the model is systematically wrong - it has learned environment-dependent features rather than true node-specific hardware signatures. The node fingerprint (driven by subtle manufacturing differences in the nRF52840 boards) is too weak to survive per-series normalisation and the domain shift between environments.

**The contrast between Strategy 1 and Strategy 2** is the central scientific result: features that appear highly discriminative within the same data distribution may fail completely when the deployment environment changes.

---

## Output Files

After running all cells, the following files are saved in the notebook's directory:

| File | Contents |
|------|----------|
| `confusion_matrices_strategy1.png` | 2x2 grid of confusion matrices for Strategy 1 |
| `confusion_matrices_strategy2.png` | 2x2 grid of confusion matrices for Strategy 2 |
| `results_comparison.png` | Bar chart comparing all 32 experiments |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'torchinfo'` | Run `pip install torchinfo` |
| `No CSV files found` | Ensure the `data_folder/` folder is in the same directory as the notebook |
| Results differ between runs | Check that Cell 1 ran successfully (seeds must be set before any training) |
| Out of memory on GPU | Reduce batch size in `make_loaders()` from 64 to 32, or use CPU |
