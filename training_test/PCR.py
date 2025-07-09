import os
import json
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from loader import MatPairDatasetSequentialCleaned

dataset = MatPairDatasetSequentialCleaned(r'C:\Users\36\OneDrive\PAPER\Journal\FSCV_DA\dat\FSCV_processing', crop_size=1, overlap_ratio=0.9)

all_X = []        # feature: (n_samples, 850)
all_y = []        # label: (n_samples, 2)
file_indices = [] 

for sample in dataset:
    color, label, domain = sample

    if hasattr(color, 'squeeze'):
        X = color.squeeze(0).cpu().numpy()
    else:
        X = np.squeeze(color, axis=0)
    if hasattr(label, 'squeeze'):
        y = label.squeeze(0).cpu().numpy()
    else:
        y = np.squeeze(label, axis=0)
    all_X.append(X)
    all_y.append(y)

    if hasattr(domain, 'item'):
        file_indices.append(int(domain.item()))
    else:
        file_indices.append(int(domain))

all_X = np.array(all_X)  # shape: (n_samples, 850)
all_y = np.array(all_y)  # shape: (n_samples, 2)

unique_files = sorted(set(file_indices))

# -------------------------
# 2. Outer Loop: Domain-based CV with PCA explained variance threshold
# -------------------------
outer_results = []  

for outer_file in unique_files:
    print(f"\n=== Outer Fold: Test file index {outer_file} ===")
    
    outer_test_indices = [i for i, idx in enumerate(file_indices) if idx == outer_file]
    outer_train_indices = [i for i, idx in enumerate(file_indices) if idx != outer_file]
    
    X_train = all_X[outer_train_indices]
    y_train = all_y[outer_train_indices]
    X_test  = all_X[outer_test_indices]
    y_test  = all_y[outer_test_indices]

    pca = PCA(n_components=0.99)
    lr = LinearRegression()
    pipeline = Pipeline([
        ('pca', pca),
        ('lr', lr)
    ])
    
    pipeline.fit(X_train, y_train)

    selected_components = int(pca.n_components_)
    

    y_pred_train = pipeline.predict(X_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    rmse_train = math.sqrt(train_mse)
    print(f"--> Outer Fold (Test file {outer_file}): Training RMSE = {rmse_train:.4f}")
    
    y_pred_test = pipeline.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    rmse_test = math.sqrt(test_mse)
    print(f"--> Outer Fold (Test file {outer_file}): Test RMSE = {rmse_test:.4f}")
    print(f"--> Outer Fold (Test file {outer_file}): Selected PCA Components = {selected_components}")
    
    outer_results.append({
        "outer_file": outer_file,
        "train_rmse": rmse_train,
        "test_rmse": rmse_test,
        "selected_components": selected_components
    })


avg_outer_test_rmse = (sum([fold['test_rmse'] for fold in outer_results]) / len(outer_results)
                       if outer_results else float('nan'))
results = {
    "explained_variance_threshold": 0.99,
    "outer_results": outer_results,
    "avg_outer_test_rmse": avg_outer_test_rmse
}

results_folder = "results/invitro_probe"
os.makedirs(results_folder, exist_ok=True)
results_path = os.path.join(results_folder, "results_PCR.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {results_path}")
