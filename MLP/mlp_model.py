import random
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BaseModel(nn.Module):
    def __init__(self, input_dim=9351):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.type_extract = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        self.tense = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )

        self.type_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=4),
        )
        self.polarity_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=3),
        )
        self.tense_classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=512, out_features=3),
        )
        self.certainty_classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2),
        )
            
    def forward(self, x):
        x1 = self.feature_extract(x)
        x2 = self.tense(x)
        x3 = self.type_extract(x)
        # ?????? ??????, ??????, ??????, ???????????? ?????? ??????
        type_output = self.type_classifier(x3)
        polarity_output = self.polarity_classifier(x1)
        tense_output = self.tense_classifier(x2)
        certainty_output = self.certainty_classifier(x1)
        return type_output, polarity_output, tense_output, certainty_output
