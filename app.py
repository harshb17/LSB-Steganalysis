from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import pickle
from scipy.stats import entropy

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# --------------------
# Model definitions
# --------------------
class BinaryDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(196614, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(192, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(96, 2))
    def forward(self, x): return self.net(x)

class MultiDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(196614, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(192, 96), nn.BatchNorm1d(96), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(96, 4))
    def forward(self, x): return self.net(x)
