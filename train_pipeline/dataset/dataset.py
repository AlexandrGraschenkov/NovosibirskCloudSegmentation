import math
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import random
import numpy as np
from sklearn.utils import class_weight

class PointsCloudDataset(Dataset):
    def __init__(self, csv_file, extra_features_csv_file=None, transform=None, verbose=False):
        if verbose: print("Dataset: Read CSV")
        self.points_frame = pd.read_csv(csv_file)
        if verbose: print("Dataset: Read finished")
        
        if extra_features_csv_file:
            if verbose: print("Dataset: Read extra features CSV")
            X_extra = pd.read_csv(extra_features_csv_file, header=None)
            if verbose: print("Dataset: Read finished")
            X_extra.columns = X_extra.columns.astype(str) # иначе дальше ломается при проверке имени
            self.points_frame = pd.concat([self.points_frame, X_extra], axis=1)
        
        self.points_frame = self.points_frame.dropna()

        self.transform = transform
        drop_cols = ["id", "Easting", "Northing", "Height"]
        if 'Class' in self.points_frame.columns:
            self.y = self.points_frame["Class"]
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
            self.y = pd.get_dummies(self.y)
            drop_cols.append("Class")
        else:
            self.y = None
        self.X = self.points_frame.drop(drop_cols, axis=1)
        y_shape = self.y.shape if self.y is not None else "-"
        if verbose: print(f"X: {self.X.shape}; y: {y_shape}")

        self.idxs_to_rotate = []
        # выбираем все фичи без магнитуды
        for idx, col in enumerate(self.X.columns):
            if "mag" not in col.lower() and ("X" in col or "Y" in col or "Z" in col):
                self.idxs_to_rotate.append(idx)

        # для быстрого доступа к памяти
        if verbose: print(f"To CUDA")
        self.X = torch.tensor(self.X.values.astype(np.float32)).to("cuda")
        if self.y is not None:
            self.y = torch.tensor(self.y.values.astype(np.float32)).to("cuda")
        else:
            self.def_y = torch.tensor(np.array([0, 0, 0, 0, 0, 0], dtype=float))
        if verbose: print(f"Dataset Done!")

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        features = self.X[idx]
        if self.y is None:
            return (features, self.def_y)
        else:
            targets = self.y[idx]
            return (features, targets)
            
        # if self.transform and random.random() > 0.5:
        #     # рандомим угол от -30 до 30 градусов
        #     angleRadians = random.uniform(-math.pi / 6, math.pi / 6)
        #     # вертим вектора на Z
        #     features = self.rotate(features, angleRadians)


    def rotate(self, data: list, angleRadians: float) -> list:
        c = math.cos(angleRadians)
        s = math.sin(angleRadians)
        for i in self.idxs_to_rotate[::3]:
            data[i], data[i + 1] = (
                data[i] * c - data[i + 1] * s,
                data[i] * s + data[i + 1] * c,
            )
        return data


if __name__ == "__main__":
    train="/home/anvar/Novosib/temp_train_ds.csv"
    # train_extra_features="/home/inna/Documents/gra_alex/novo_hackathon/train_dataset_train.csv_result_2"
    train_ds = PointsCloudDataset(train, transform=True, verbose=True)
    print(train_ds.class_weights)
    # print(train_ds[0])
    # print(train_ds[0][0].shape)
    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # print(next(iter(train_loader)))
