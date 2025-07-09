import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io

class MatPairDatasetSequentialCleaned(Dataset):

    def __init__(self, base_folder, crop_size=150, overlap_ratio=0.5):

        self.data_pairs = []
        self.crop_size = crop_size
        self.stride = max(1, int(crop_size * (1 - overlap_ratio)))  
        self.index_mapping = []

        # 데이터 로드
        for root, dirs, files in os.walk(base_folder):
            mat_files = [f for f in files if f.endswith('.mat')]
            color_files = [f for f in mat_files if f.endswith('Color.mat')]
            label_files = [f for f in mat_files if f.endswith('_L.mat')]

            for color_file in color_files:
                label_file = color_file.replace('Color.mat', 'Color_L.mat')
                if label_file in label_files:
                    color_path = os.path.join(root, color_file)
                    label_path = os.path.join(root, label_file)

                    color_data = scipy.io.loadmat(color_path)['d'] / 1000  # normalize to [-0.5,0.5] scale
                    label_data = scipy.io.loadmat(label_path)['L'] / 1000  # normalize to [0,0.5] scale

                    color_data = color_data - color_data[0]

                    # remove rows with NaN values
                    valid_rows = ~np.any(np.isnan(label_data), axis=1)
                    color_data = color_data[valid_rows, :]
                    label_data = label_data[valid_rows, :]

                    T = color_data.shape[0]  # 전체 시간 길이

                    # 순차적 크롭 인덱스 생성
                    crop_indices = [
                        (len(self.data_pairs), start_idx)
                        for start_idx in range(0, T - crop_size + 1, self.stride)
                    ]
                    self.data_pairs.append((color_data, label_data))
                    self.index_mapping.extend(crop_indices)
                else:
                    print(f"[WARN] Not exist _L.mat file with {color_file}.")

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):

        file_idx, start_idx = self.index_mapping[idx]
        color_data, label_data = self.data_pairs[file_idx]

        color_crop = color_data[start_idx : start_idx + self.crop_size, :]  # [T x V]
        label_crop = label_data[start_idx : start_idx + self.crop_size, :]  # [T x 2]
        domain_crop = np.array(file_idx)  # 도메인 정보 (여기서는 파일 인덱스로 사용)

        # Numpy -> Tensor 변환
        color_crop = torch.tensor(color_crop, dtype=torch.float32)
        label_crop = torch.tensor(label_crop, dtype=torch.float32)
        domain_crop = torch.tensor(domain_crop, dtype=torch.float32)

        return color_crop, label_crop, domain_crop

class MatSimulationDatasetSequential(Dataset):

    def __init__(self, file_path, crop_size=150, overlap_ratio=0.5):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] File {file_path} is not exist.")
        
        mat_contents = scipy.io.loadmat(file_path)
        self.data = mat_contents['data']*1e5    # shape: (5000, 850)
        self.label = mat_contents['Label']*1e11   # shape: (5000, 2)

        self.crop_size = crop_size
        self.stride = max(1, int(crop_size * (1 - overlap_ratio))) 
        
        T = self.data.shape[0]
        self.index_mapping = [start_idx for start_idx in range(0, T - crop_size + 1, self.stride)]

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):

        start_idx = self.index_mapping[idx]
        data_crop = self.data[start_idx : start_idx + self.crop_size, :]
        label_crop = self.label[start_idx : start_idx + self.crop_size, :]
        
        # Numpy to Tensor
        data_crop = torch.tensor(data_crop, dtype=torch.float32)
        label_crop = torch.tensor(label_crop, dtype=torch.float32)
        
        return data_crop, label_crop, start_idx

