import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from loader import MatPairDatasetSequentialCleaned

source_dataset = MatPairDatasetSequentialCleaned(
    r'C:\Users\36\OneDrive\PAPER\Journal\FSCV_DA\dat\FSCV_processing',
    crop_size=1, overlap_ratio=0.9
)

all_X = []  # (n_samples, 850)
all_y = []  # (n_samples, 2)

for sample in source_dataset:
    color, label, _ = sample
    X = color.squeeze(0).cpu().numpy() if hasattr(color, 'squeeze') else np.squeeze(color, axis=0)
    y = label.squeeze(0).cpu().numpy() if hasattr(label, 'squeeze') else np.squeeze(label, axis=0)
    all_X.append(X)
    all_y.append(y)

all_X = np.array(all_X)
all_y = np.array(all_y)

X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.3, random_state=42)

alpha_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
l1_ratio_list = [0.01, 0.1, 0.5, 0.9]
best_params = None
best_cv_rmse = float('inf')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for alpha in alpha_list:
    for l1_ratio in l1_ratio_list:
        cv_rmses = []
        for train_idx, val_idx in kf.split(X_train):
            X_inner_train, X_inner_val = X_train[train_idx], X_train[val_idx]
            y_inner_train, y_inner_val = y_train[train_idx], y_train[val_idx]
            
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            model.fit(X_inner_train, y_inner_train)
            y_pred_val = model.predict(X_inner_val)
            rmse = math.sqrt(mean_squared_error(y_inner_val, y_pred_val))
            cv_rmses.append(rmse)
        
        avg_rmse = np.mean(cv_rmses)
        if avg_rmse < best_cv_rmse:
            best_cv_rmse = avg_rmse
            best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

print(f"Best hyperparameters from nested CV: {best_params} with CV RMSE: {best_cv_rmse:.4f}")


final_model = MultiTaskElasticNet(alpha=best_params["alpha"],
                                  l1_ratio=best_params["l1_ratio"],
                                  max_iter=10000)
final_model.fit(X_train, y_train)
y_pred_source = final_model.predict(X_test)
source_rmse = math.sqrt(mean_squared_error(y_test, y_pred_source))
print(f"Final Source Test RMSE: {source_rmse:.4f}")

target_dataset = MatPairDatasetSequentialCleaned(
    r'C:\Users\36\OneDrive\PAPER\Journal\FSCV_DA\dat\FSCV_pH',
    crop_size=1, overlap_ratio=0.9
)

target_file_indices = []
for sample in target_dataset:
    _, _, file_id = sample
    target_file_indices.append(file_id)
target_file_indices = np.array(target_file_indices)
target_unique_files = np.unique(target_file_indices)
print(f"Unique test files found: {target_unique_files}")

outer_results = []
results_folder = "results/invitro_pH"
os.makedirs(results_folder, exist_ok=True)
model_folder = os.path.join(results_folder, "model_SSAE_2")
os.makedirs(model_folder, exist_ok=True)

for outer_file in target_unique_files:
    print(f"\n=== Outer Fold: Test file index {outer_file} (target dataset) ===")
    outer_test_indices = [i for i, idx in enumerate(target_file_indices) if idx == outer_file]
    outer_train_indices = [i for i, idx in enumerate(target_file_indices) if idx != outer_file]
    
    outer_train_subset_target = Subset(target_dataset, outer_train_indices)
    outer_test_subset_target  = Subset(target_dataset, outer_test_indices)
    
    target_train_size = int(0.5 * len(outer_test_subset_target))
    target_test_size = len(outer_test_subset_target) - target_train_size
    target_train_subset, target_test_subset = torch.utils.data.random_split(
        outer_test_subset_target, [target_train_size, target_test_size]
    )
    
    file_X = []
    file_y = []
    for i in range(len(target_test_subset)):
        sample = target_test_subset[i]
        color, label, _ = sample
        X = color.squeeze(0).cpu().numpy() if hasattr(color, 'squeeze') else np.squeeze(color, axis=0)
        y = label.squeeze(0).cpu().numpy() if hasattr(label, 'squeeze') else np.squeeze(label, axis=0)
        file_X.append(X)
        file_y.append(y)
    file_X = np.array(file_X)
    file_y = np.array(file_y)
    
    y_pred_file = final_model.predict(file_X)
    file_mse = mean_squared_error(file_y, y_pred_file)
    file_rmse = math.sqrt(file_mse)
    print(f"Test File {outer_file} RMSE: {file_rmse:.4f}")
    
    outer_results.append({
        "test_file": str(outer_file),
        "file_rmse": file_rmse,
        "num_samples": len(target_test_subset)
    })

final_results = {
    "best_hyperparameters": best_params,
    "nested_cv_rmse": best_cv_rmse,
    "source_test_rmse": source_rmse,
    "outer_results": outer_results
}

results_path = os.path.join(results_folder, "results_ElasticNet_final.json")
with open(results_path, "w") as f:
    json.dump(final_results, f, indent=4)
print(f"Final results saved to {results_path}")
