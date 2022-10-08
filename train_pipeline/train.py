import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import argparse
from mmcv import Config
from sklearn import metrics
from utils import get_predictions
from model import NN
from dataset.dataset import PointsCloudDataset
from torch.utils.data import DataLoader
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    return args


def save_predicts(loader, model):
    probabilities, true = get_predictions(loader, model, device=DEVICE)
    true_labels = np.argmax(true, axis=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    print(f"confusion_matrix:\n{confusion_matrix(true_labels, predicted_labels)}")
    print(f"recall_score: {recall_score(true_labels, predicted_labels, average='micro', zero_division=True)}")
    print(f"precision_score: {precision_score(true_labels, predicted_labels, average='macro', zero_division=True)}")
    print(f"accuracy_score: {accuracy_score(true_labels, predicted_labels)}")

    # приводим маппинг классов к изначальному виду
    pred_labels_new = []
    for i in predicted_labels:
        if i == 0 or i == 1:
            pred_labels_new.append(i)
        elif i in [2, 3, 4]:
            pred_labels_new.append(i + 1)
        elif i == 5:
            pred_labels_new.append(64)

    # сохраняем в csv
    last_row = pd.DataFrame({'pred_class': pred_labels_new})
    last_row.to_csv(config.paths["save_preds"], index=False)


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config_path)
    paths = config.paths
    extra_features = paths.train_extra_features if hasattr(paths, 'train_extra_features') else None
    train_ds = PointsCloudDataset(paths.train, extra_features_csv_file=extra_features, transform=True, verbose=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    input_size = len(train_ds[0][0])
    DEVICE = config.device
    model = NN(input_size=input_size, hidden_dim=config.nn["hidden_dims"]).to(DEVICE)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        probabilities, true = get_predictions(loader=train_loader, model=model, is_test=False, device=DEVICE)
        print(f"VALIDATION ROC: {metrics.roc_auc_score(true, probabilities)}")
        print("Recall score",
              recall_score(np.argmax(true, axis=1), np.argmax(probabilities, axis=1), average='micro',
                           zero_division=True))

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = torch.tensor(targets, dtype=torch.float).to(DEVICE)
            scores = model(data.float())
            loss = loss_fn(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), config.paths["save_nn"])
    save_predicts(train_loader, model)
