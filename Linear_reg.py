import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"https://raw.githubusercontent.com/Paras-bakshi/Laptop-Price-Predictor/main/laptop_data.csv")
data.head()
data_a = data[['Inches','Price']]
x = data_a[['Inches']].values
y = data_a[['Price']].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .2,random_state = 42)
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(x_test, dtype=torch.float32)

class Linearregress(nn.Module):
  def __init__(self,input,output):
    super(Linearregress,self).__init__()
    self.linear = nn.Linear(input,output)
  
  def forward(self,x):
    return self.linear(x)

mod = Linearregress(1,1)
criterion = nn.MSELoss()
optim = torch.optim.Adam(params = mod.parameters(), lr = 0.001)
epoch = 1000
losses = []
for i in range(0,epoch):
  output = mod(X_train_tensor)
  loss = criterion(output,y_train_tensor)
  losses.append(loss.item())
  optim.zero_grad()
  loss.backward()
  optim.step()
