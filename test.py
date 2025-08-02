import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

y = torch.arange(100)
x = y.clone()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

train = TensorDataset(x_train, y_train)
test = TensorDataset(x_test, y_test)

train_loader = DataLoader(train, batch_size=10, shuffle=True)
test_loader = DataLoader(test, batch_size=4, shuffle=False)
aa = list(test_loader)
