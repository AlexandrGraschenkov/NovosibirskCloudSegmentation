import math
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np


class PointsCloudDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.points_frame = pd.read_csv(csv_file)
        self.points_frame = self.points_frame.dropna()
        self.transform = transform
        self.y = self.points_frame["Class"]
        self.y = pd.get_dummies(self.y)
        self.X = self.points_frame.drop(["Class", "id", "Easting", "Northing", "Height"], axis=1)

        self.idxs_to_rotate = []
        # выбираем все фичи без магнитуды
        for idx, col in enumerate(self.X.columns):
            if "mag" not in col.lower() and ("X" in col or "Y" in col or "Z" in col):
                self.idxs_to_rotate.append(idx)

    def __len__(self):
        return len(self.points_frame)

    def __getitem__(self, idx):
        features = self.X.iloc[idx]
        targets = self.y.iloc[idx]
        if self.transform and random.random() > 0.5:
            # рандомим угол от -30 до 30 градусов
            angleRadians = random.uniform(-math.pi / 6, math.pi / 6)
            # вертим вектора на Z
            features = self.rotate(features, angleRadians)

        return np.array(features), np.array(targets)

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
    train_ds = PointsCloudDataset("/home/anvar/Novosib/temp_train_ds.csv", transform=True)
    print(train_ds[0])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # print(next(iter(train_loader)))
