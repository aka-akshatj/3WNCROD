# MAT Dataset Support

The 3WNCROD project supports MATLAB `.mat` files for outlier detection experiments.

## Using MAT Files

### 1. **Prepare Your MAT Files**

Place your MAT files in the `Datasets/` folder:
```
Code/
├── Datasets/
│   ├── annthyroid.mat          # Original .mat files
│   ├── mammography.mat
│   ├── thyroid.mat
│   ├── vertebral.mat           # New .mat datasets
│   ├── pima.mat
│   ├── ionosphere.mat
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
    
    # New .mat datasets
    'vertebral',
    'pima',
    'ionosphere',
    'cardio',
    'satellite',
    'shuttle',
]
```

### 3. **Run the Pipeline**

```bash
cd Code
python run_pyod.py
```

The system will:
- Automatically detect `.mat` files
- Extract the data matrix
- Normalize all features to [0,1]
- Run all PyOD algorithms including AutoEncoder

---

## MAT File Requirements

### 1. **Data Structure**
Your `.mat` file should contain a 2D array:
- **Rows**: Samples/observations
- **Columns**: Features/attributes

```matlab
% MATLAB example
data = [1.1, 2.5, 3.7, 1.0;
        2.2, 1.8, 4.2, 0.5;
        3.3, 2.1, 3.9, 0.8];

save('mydata.mat', 'data');
```

### 2. **Variable Naming**
The variable name should match the dataset filename (optional):
```matlab
annthyroid = [...];  % Variable name = file name (best practice)
save('annthyroid.mat', 'annthyroid');
```

If variable name doesn't match, the system auto-detects the first non-system variable.

### 3. **Data Type**
- **Recommended**: `double` (float64)
- **Supported**: `single` (float32), `uint8`, `int32`, etc.
- **Not supported**: Complex numbers, cell arrays, structs (nested)

### 4. **Missing Values**
Replace NaN or Inf values before saving:
```matlab
% Handle missing values
data(isnan(data)) = 0;           % Replace NaN with 0
data(isinf(data)) = max(data);   % Replace Inf with max value

save('clean_data.mat', 'data');
```

### 5. **Categorical Features**
For categorical data, convert to integers first:
```matlab
% Example: Convert categorical to numerical
categories = categorical({'A', 'B', 'C', 'A', 'B'});
numeric_categories = int32(categories) - 1;  % 0, 1, 2 encoding

data = [numeric_categories; other_features];
save('mixed_data.mat', 'data');
```

---

## Creating MAT Files

### From MATLAB

```matlab
% Simple case
data = readmatrix('data.csv');
save('dataset.mat', 'data');

% With variable name
mydata = readmatrix('data.csv');
save('mydataset.mat', 'mydata');

% With preprocessing
data = readmatrix('raw_data.csv');
data = data(~any(isnan(data), 2), :);  % Remove NaN rows
save('clean_data.mat', 'data');
```

### From Python

```python
import scipy.io as sio
import numpy as np

# Create data
data = np.random.randn(100, 10)

# Save as .mat file
sio.savemat('dataset.mat', {'dataset': data})

# From pandas DataFrame
import pandas as pd
df = pd.read_csv('data.csv')
data = df.values
sio.savemat('dataset.mat', {'dataset': data})

# From NumPy array
array = np.array([[1, 2, 3], [4, 5, 6]])
sio.savemat('array_data.mat', {'array_data': array})
```

### From CSV

```python
import pandas as pd
import scipy.io as sio
import numpy as np

# Read CSV
df = pd.read_csv('data.csv')

# Convert categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Convert to numpy
data = df.values.astype(np.float64)

# Save as MAT
sio.savemat('dataset.mat', {'dataset': data})
```

### From Excel

```python
import pandas as pd
import scipy.io as sio
import numpy as np

# Read Excel
df = pd.read_excel('data.xlsx')

# Convert categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Convert to MAT
data = df.values.astype(np.float64)
sio.savemat('dataset.mat', {'dataset': data})
```

---

## MAT File Specifications

### File Size
- **Small**: < 1 MB (< 10,000 samples)
- **Medium**: 1-100 MB (10,000 - 1,000,000 samples)
- **Large**: > 100 MB (> 1,000,000 samples)

MAT format is efficient for binary storage compared to CSV.

### Compression
MATLAB can compress .mat files:
```matlab
save('compressed.mat', 'data', '-v7.3');  % Saves with compression
```

### Variable Storage
One `.mat` file can contain multiple variables:
```matlab
% Save multiple variables
var1 = [...];
var2 = [...];
save('data.mat', 'var1', 'var2');

% The system will extract the first non-system variable
```

---

## Loading MAT Files in 3WNCROD

The system automatically:

1. **Detects MAT format**
   - Checks file extension `.mat`
   
2. **Extracts data**
   - Reads all variables
   - Selects first non-system variable (excluding `__*` variables)
   - Converts to numpy array

3. **Preprocesses data**
   - Handles NaN and Inf values
   - Converts to float64
   - Normalizes to [0,1] range

4. **Runs algorithms**
   - IForest, LOF, COPOD, HBOS, KNN
   - AutoEncoder
   - Custom WNCROD

---

## Supported MAT Versions

| Version | Support | Notes |
|---------|---------|-------|
| MAT v4 | ✅ Yes | Legacy MATLAB format |
| MAT v5 | ✅ Yes | Standard format |
| MAT v7.3 | ✅ Yes | Large file support |

All versions are automatically detected and loaded.

---

## Troubleshooting

### Error: "File not found"
```
ERROR: File not found: ./Datasets/mydata.mat
```
**Solution**: Check file exists in `Code/Datasets/` folder

### Error: "No data variable found"
```
ERROR: Unable to extract data from MAT file
```
**Solution**: MAT file contains no regular variables. Check file is valid:
```python
import scipy.io as sio
mat_data = sio.loadmat('file.mat')
print(mat_data.keys())  # See all variables
```

### Error: "Data has NaN values"
```
WARNING: Data contains NaN values
```
**Solution**: Replace NaN before saving:
```matlab
data(isnan(data)) = 0;
save('clean.mat', 'data');
```

### Error: "Incompatible shape"
```
ERROR: Data must be 2D array (samples × features)
```
**Solution**: Reshape your data to 2D:
```python
import scipy.io as sio
import numpy as np

data = np.array([...])
if data.ndim == 1:
    data = data.reshape(-1, 1)  # Convert to 2D

sio.savemat('dataset.mat', {'dataset': data})
```

---

## Performance Tips

1. **Use MAT over CSV for large datasets** (> 100k samples)
   - MAT is binary and loads faster
   - CSV requires parsing and conversion

2. **Pre-normalize if possible** (optional)
   - System normalizes automatically
   - But faster if already normalized

3. **Remove unnecessary variables**
   ```matlab
   % Bad: file contains many unused variables
   % Good: save only data matrix
   save('dataset.mat', 'data');
   ```

4. **Use compression for storage**
   ```matlab
   save('dataset.mat', 'data', '-v7.3');  % Compressed
   ```

---

## Comparison: MAT vs CSV

| Aspect | MAT | CSV |
|--------|-----|-----|
| File Size | Smaller (binary) | Larger (text) |
| Load Speed | Fast | Slower |
| Human Readable | No | Yes |
| Format Conversion | Need MATLAB/SciPy | Easy (text editor) |
| Compatibility | MATLAB/Python/R | Universal |
| Large Files | Efficient | Memory-heavy |
| Mixed Types | Yes | Yes |

**Recommendation**: 
- Use **MAT** for: Large datasets, frequent processing
- Use **CSV** for: Easy sharing, data inspection, small datasets

---

## Example Workflow

### Step 1: Create MAT file from data
```python
import pandas as pd
import scipy.io as sio

df = pd.read_csv('raw_data.csv')
df = df.dropna()  # Remove missing values

data = df.values.astype(np.float64)
sio.savemat('Code/Datasets/mydata.mat', {'mydata': data})
print("✓ Saved: Code/Datasets/mydata.mat")
```

### Step 2: Add to run_pyod.py
```python
datasets = [
    'annthyroid',
    'mydata',  # Your new dataset
]
```

### Step 3: Run experiments
```bash
python run_pyod.py
```

### Step 4: Check results
```
Results in: output_logs/pyod_output_<timestamp>.txt
```

---

## Quick Reference

| Task | Code |
|------|------|
| Save array | `sio.savemat('file.mat', {'var': array})` |
| Load file | `mat = sio.loadmat('file.mat')` |
| Check contents | `print(mat.keys())` |
| Get data | `data = mat['varname']` |
| Remove NaN | `data[np.isnan(data)] = 0` |
| Convert category | `pd.Categorical(data).codes` |

---

## Contact & Support

For issues with MAT files:
1. Verify file format: `python -c "import scipy.io as sio; print(sio.loadmat('file.mat').keys())"`
2. Check data shape: `print(array.shape)`
3. Ensure 2D format: `array.ndim == 2`

Questions? Check `CSV_USAGE_GUIDE.md` for similar workflow with CSV files.
