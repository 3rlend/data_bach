import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# Endre denne til mappen med CSV-filene som skal analyseres
folder = r"sti\til\data"
# ============================================================

window_size = 832
step = 416

# Modellarkitektur (må være identisk med den trente modellen)
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Last inn modell og normaliseringsparametre
model = CNN1D()
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
mean = np.load("mean.npy")
std  = np.load("std.npy")

# Les og vindusoppdel data
windows = []
for fname in [f for f in os.listdir(folder) if f.endswith(".csv")]:
    df = pd.read_csv(os.path.join(folder, fname))
    data = df[["x", "y", "z"]].values.astype(np.float32)
    for start in range(0, len(data) - window_size, step):
        windows.append(data[start:start + window_size])

X = torch.tensor(np.array(windows).transpose(0, 2, 1), dtype=torch.float32)
X = (X - mean) / std  # normaliser

# Prediksjon
with torch.no_grad():
    preds = model(X).argmax(dim=1)

fast = (preds == 0).sum().item()
løs  = (preds == 1).sum().item()
total = len(preds)

print(f"Faste vinduer:  {fast} ({fast/total:.1%})")
print(f"Løse vinduer:   {løs} ({løs/total:.1%})")
print(f"\n>>> Samlet vurdering: {'FAST' if fast > løs else 'LØS'} <<<")