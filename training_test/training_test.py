import os
import json
import math
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import MatSimulationDatasetSequential
from torch.utils.data import random_split
from net import DSN_Network_linear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    outer_epochs = 1000
    batch_size = 256
    learning_rate = 1e-4
    domain_lr = 1e-4
    
    lambda_recon = 1.0
    lambda_domain = 1.0
    
    supervised_criterion = nn.MSELoss(reduction='mean')
    reconstruction_criterion = nn.MSELoss(reduction='mean')
    domain_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    model_type = 'ablation_dance'
        
    results_folder = r'C:\Users\36\OneDrive\PAPER\Journal\FSCV_DA\results\insilico_probe_change'
    os.makedirs(results_folder, exist_ok=True)
    model_folder = os.path.join(results_folder, f"models_{model_type}")
    os.makedirs(model_folder, exist_ok=True)
    
    base_folder = r'\simulation_data\probe_change' # paths to the dataset
    
    source_file = os.path.join(base_folder, "simulation_probe_0.00.mat")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"[ERROR] Not exist source files: {source_file}")
    
    ds_source = MatSimulationDatasetSequential(source_file, crop_size=1, overlap_ratio=0.5)
    n_source = len(ds_source)
    
    source_train_size = int(0.5 * n_source)
    source_val_size = n_source - source_train_size
    source_train_subset, source_val_subset = random_split(ds_source, [source_train_size, source_val_size])
    
    target_values = [0.10]
    n_splits = 1
    mean_rmse_list = []
    std_rmse_list = []
    raw_rmse_matrix = []
    outer_results = []
    
    for target_value in target_values:
        target_file = os.path.join(base_folder, f"simulation_probe_{target_value:.2f}.mat")
        if not os.path.exists(target_file):
            print(f"[WARN] Not exist source files: {target_file}")
            continue
        print(f"\n=== Target file : {target_file} (domain=1) ===")
        
        if target_value == 0.00:
            ds_target = MatSimulationDatasetSequential(target_file, crop_size=1, overlap_ratio=0.5)
            n_target = len(ds_target)
            train_size = int(0.5 * n_target)
            val_size = n_target - train_size
            target_train_subset, target_val_subset = random_split(ds_target, [train_size, val_size])
            used_source_train_subset = target_train_subset
            source_loader = DataLoader(used_source_train_subset, batch_size=batch_size, shuffle=True)
            target_loader = DataLoader(target_train_subset, batch_size=batch_size, shuffle=True)
            target_eval_loader = DataLoader(target_val_subset, batch_size=batch_size, shuffle=False)
        else:
            target_dataset = MatSimulationDatasetSequential(target_file, crop_size=1, overlap_ratio=0.5)
            n_target = len(target_dataset)
            target_train_size = int(0.5 * n_target)
            target_val_size = n_target - target_train_size
            target_train_subset, target_val_subset = random_split(target_dataset, [target_train_size, target_val_size])
            used_source_train_subset = source_train_subset
            source_loader = DataLoader(used_source_train_subset, batch_size=batch_size, shuffle=True)
            target_loader = DataLoader(target_train_subset, batch_size=batch_size, shuffle=True)
            target_eval_loader = DataLoader(target_val_subset, batch_size=batch_size, shuffle=False)
        
        if target_value != 0.00:
            source_eval_loader = DataLoader(source_val_subset, batch_size=batch_size, shuffle=False)
        
        source_size = len(used_source_train_subset)
        target_size = len(target_train_subset)
        target_ratio = target_size / source_size if source_size > 0 else 0
        target_batches = list(target_loader)
        num_target_batches = len(target_batches)
        target_update_freq = round(1 / target_ratio) if target_ratio > 0 else 1000000
        print(f"Target update frequency: {target_update_freq} (target_ratio: {target_ratio:.4f})")
        
        eval_rmse_target_list = []
        for split in range(n_splits):
            if target_value == 0.00:
                ds_target = MatSimulationDatasetSequential(target_file, crop_size=1, overlap_ratio=0.5)
                n_target = len(ds_target)
                train_size = int(0.5 * n_target)
                val_size = n_target - train_size
                _, target_val_subset = random_split(ds_target, [train_size, val_size])
                target_eval_loader = DataLoader(target_val_subset, batch_size=batch_size, shuffle=False)
            else:
                n_target = len(target_dataset)
                target_train_size = int(0.5 * n_target)
                target_val_size = n_target - target_train_size
                _, target_val_subset = random_split(target_dataset, [target_train_size, target_val_size])
                target_eval_loader = DataLoader(target_val_subset, batch_size=batch_size, shuffle=False)
            
            use_private = True
            model = DSN_Network_linear(input_dim=850, shared_dim=60, private_dim=20, output_dim=2,
                                num_domains=2, grl_lambda=1.0,
                                use_private_encoder=use_private).to(device)
            domain_classifier_params = list(model.domain_classifier.parameters())
            other_params = list(model.shared_encoder.parameters())
            if use_private:
                other_params += list(model.private_encoder.parameters())
            other_params += list(model.fc.parameters())
            other_params += list(model.decoder.parameters())
            
            optimizer = optim.Adam([
                {"params": other_params, "lr": learning_rate},
                {"params": domain_classifier_params, "lr": domain_lr}
            ], betas=(0.9, 0.999))
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
            
            # training loop
            for epoch in range(outer_epochs):
                model.train()
                total_loss_epoch = 0.0
                batch_count = 0
                
                current_grl_lambda = 1.0 * (2 / (1 + math.exp(-10 * (epoch / outer_epochs))) - 1)
                current_grl_lambda_tensor = torch.tensor(current_grl_lambda, device=device)
                
                for i, (data_s, label_s, _) in enumerate(source_loader):
                    data_s = data_s.to(device)
                    label_s = label_s.to(device)
                    domain_s = torch.zeros(data_s.size(0), device=device, dtype=torch.long)
                    
                    y_pred_s, x_recon_s, domain_pred_s, _ = model(data_s, current_grl_lambda_tensor)
                    valid_mask_s = ~torch.isnan(label_s)
                    sup_loss = supervised_criterion(y_pred_s[valid_mask_s], label_s[valid_mask_s]) if valid_mask_s.sum() > 0 else 0.0
                    recon_loss_s = reconstruction_criterion(x_recon_s, data_s)
                    
                    domain_s_expanded = domain_s.unsqueeze(1).expand(-1, data_s.size(1)).reshape(-1)
                    dom_loss_s = domain_criterion(domain_pred_s.view(-1, 2), domain_s_expanded)
                    
                    if i % target_update_freq == 0 and num_target_batches > 0:
                        tgt_batch = target_batches[i % num_target_batches]
                        data_t, _, _ = tgt_batch
                        data_t = data_t.to(device)
                        domain_t = torch.ones(data_t.size(0), device=device, dtype=torch.long)
                        
                        _, x_recon_t, domain_pred_t, _ = model(data_t, current_grl_lambda_tensor)
                        recon_loss_t = reconstruction_criterion(x_recon_t, data_t)
                        
                        domain_t_expanded = domain_t.unsqueeze(1).expand(-1, data_t.size(1)).reshape(-1)
                        dom_loss_t = domain_criterion(domain_pred_t.view(-1, 2), domain_t_expanded)
                        
                        total_recon_loss = (recon_loss_s + recon_loss_t) / (data_s.numel() + data_t.numel())
                        total_domain_loss = (dom_loss_s + dom_loss_t) / 2.0
                    else:
                        total_recon_loss = recon_loss_s / data_s.numel()
                        total_domain_loss = dom_loss_s
                    
                    loss = sup_loss + lambda_recon * total_recon_loss + lambda_domain * total_domain_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss_epoch += loss.item()
                    batch_count += 1
                scheduler.step()
                
                if epoch % 10 == 0:
                    model.eval()
                    total_sup_loss_tgt = 0.0
                    total_batches_tgt = 0
                    with torch.no_grad():
                        for data, label, _ in target_eval_loader:
                            data = data.to(device)
                            label = label.to(device)
                            y_pred, _, _, _ = model(data)
                            valid_mask = ~torch.isnan(label)
                            if valid_mask.sum() == 0:
                                continue
                            loss_tgt = supervised_criterion(y_pred[valid_mask], label[valid_mask])
                            total_sup_loss_tgt += loss_tgt.item()
                            total_batches_tgt += 1
                    avg_sup_loss_tgt = total_sup_loss_tgt / total_batches_tgt if total_batches_tgt > 0 else float('nan')
                    rmse_target = math.sqrt(avg_sup_loss_tgt) if not math.isnan(avg_sup_loss_tgt) else float('nan')
                    print(f"Split {split+1}, Epoch {epoch}: Validation RMSE = {rmse_target:.4f}")
                    model.train()
            
            model.eval()
            total_sup_loss_tgt = 0.0
            total_batches_tgt = 0
            with torch.no_grad():
                for data, label, _ in target_eval_loader:
                    data = data.to(device)
                    label = label.to(device)
                    y_pred, _, _, _ = model(data)
                    valid_mask = ~torch.isnan(label)
                    if valid_mask.sum() == 0:
                        continue
                    loss_tgt = supervised_criterion(y_pred[valid_mask], label[valid_mask])
                    total_sup_loss_tgt += loss_tgt.item()
                    total_batches_tgt += 1
                    
            
            avg_sup_loss_tgt = total_sup_loss_tgt / total_batches_tgt if total_batches_tgt > 0 else float('nan')
            rmse_target = math.sqrt(avg_sup_loss_tgt) if not math.isnan(avg_sup_loss_tgt) else float('nan')
            eval_rmse_target_list.append(rmse_target)

        if len(eval_rmse_target_list) >= 1:
            final_rmse_target = np.mean(eval_rmse_target_list)
            std_rmse_target = np.std(eval_rmse_target_list)
        else:
            final_rmse_target = float('nan')
            std_rmse_target = float('nan')
        
        print(f"--> Target File {target_value}: Mean Target RMSE = {final_rmse_target:.4f}, STD = {std_rmse_target:.4f}\n")
        outer_results.append({
            "target_file": target_value,
            "mean_target_rmse": final_rmse_target,
            "std_target_rmse": std_rmse_target,
            "raw_target_rmse": eval_rmse_target_list
        })
        mean_rmse_list.append(final_rmse_target)
        std_rmse_list.append(std_rmse_target)
        raw_rmse_matrix.append(eval_rmse_target_list)
    
    avg_target_rmse = np.mean(mean_rmse_list) if mean_rmse_list else float('nan')
    results = {
        "outer_epochs": outer_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "domain_lr": domain_lr,
        "target_values": target_values,
        "mean_target_test_rmse": mean_rmse_list,
        "std_target_test_rmse": std_rmse_list,
        "raw_target_test_rmse": raw_rmse_matrix,
        "avg_target_test_rmse": avg_target_rmse,
        "outer_results": outer_results
    }
    
    results_path = os.path.join(results_folder, f"results_{model_type}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Nested CV results saved to {results_path}")
