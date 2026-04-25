# CSV Dataset Support

The 3WNCROD project now supports both `.mat` and `.csv` file formats for outlier detection!

## Using CSV Files

### 1. **Prepare Your CSV Files**

Place your CSV files in the `Datasets/` folder:
```
Code/
├── Datasets/
│   ├── annthyroid.mat          # Existing .mat files still work
│   ├── vertebral.csv           # New CSV files
│   ├── pima.csv
│   ├── ionosphere.csv
│   └── ...
```

### 2. **Add Dataset Names to run_pyod.py**

In `Code/run_pyod.py`, update the datasets list:

```python
datasets = [
    # Existing .mat datasets
    'annthyroid',
    'mammography',
    'thyroid',
    
    # New .csv datasets (no need to specify extension)
    'vertebral',      # Will auto-detect vertebral.csv
    'pima',           # Will auto-detect pima.csv
    'ionosphere',     # Will auto-detect ionosphere.csv
    'cardio',
    'satellite',
    'shuttle',
    'kddcup99'
]
```

### 3. **Run the Pipeline**

```bash
cd Code
python run_pyod.py
```

The system will:
- Automatically detect `.csv` files
- Convert categorical columns to numerical
- Normalize all features to [0,1]
- Run all PyOD algorithms including AutoEncoder

---

## Converting Datasets to CSV

### From Other Formats

#### **From .mat files:**
```python
from scipy.io import loadmat, savemat
import pandas as pd

# Load .mat file
mat_data = loadmat('dataset.mat')
data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
data = mat_data[data_key]

# Convert to CSV
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False, header=False)
```

#### **From .arff files:**
```python
from scipy.io import arff
import pandas as pd

# Load ARFF file
data, meta = arff.loadarff('dataset.arff')
df = pd.DataFrame(data)

# Convert to CSV
df.to_csv('dataset.csv', index=False, header=False)
```

#### **From .data files (KDD Cup format):**
```python
import pandas as pd

# Load raw data file
df = pd.read_csv('dataset.data', header=None)

# Convert categorical columns to numerical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Save as CSV
df.to_csv('dataset.csv', index=False, header=False)
```

#### **From Excel (.xlsx) files:**
```python
import pandas as pd

# Load Excel file
df = pd.read_excel('dataset.xlsx')

# Convert categorical columns to numerical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Save as CSV
df.to_csv('dataset.csv', index=False, header=False)
```

---

## CSV File Format Requirements

1. **No headers** - First row should be data, not column names
   ```
   1.1,2.5,3.7,1
   2.2,1.8,4.2,0
   3.3,2.1,3.9,1
   ```

2. **Numerical format** - All values must be numerical
   - Categorical values like "yes/no" will be converted to integers (0, 1, 2, ...)
   
3. **Missing values** - Replace with 0 or mean value before saving

4. **Consistent dimensions** - All rows must have the same number of columns

---

## Example: Converting a New Dataset

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('raw_data.csv')

# Handle missing values
df = df.dropna()  # or df.fillna(0)

# Convert categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Save without headers and index
df.to_csv('Datasets/mydataset.csv', index=False, header=False)

print("✓ Dataset ready at Datasets/mydataset.csv")
```

---

## How the Auto-Detection Works

The system automatically:

1. Checks if file is `.mat` or `.csv`
2. Loads appropriate format
3. Converts categorical columns to integers
4. Handles missing/infinite values
5. Normalizes to [0,1] range
6. Runs all algorithms

No manual file format specification needed!

---

## Supported Formats

| Format | Support | Auto-Detect |
|--------|---------|-------------|
| `.mat` | ✅ Yes | ✅ Yes |
| `.csv` | ✅ Yes | ✅ Yes |
| `.arff` | Manual convert | ❌ No |
| `.data` | Manual convert | ❌ No |
| `.xlsx` | Manual convert | ❌ No |
| `.json` | Manual convert | ❌ No |
| `.txt` | Manual convert | ❌ No |

---

## Tips

- **Keep CSV files in same location**: `Code/Datasets/yourdata.csv`
- **Use simple names**: No spaces, only alphanumeric and underscores
- **Large files**: CSV is generally slower than .mat; consider .mat for 100k+ rows
- **Mixed data**: Both categorical and numerical columns are supported

Questions? Check the test example with actual CSV files!
