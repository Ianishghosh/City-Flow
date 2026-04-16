import torch
from torch.utils.data import TensorDataset, DataLoader

def make_tensors(X, y):
    return torch.FloatTensor(X), torch.FloatTensor(y)

def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    train_ds = TensorDataset(*make_tensors(X_train, y_train))
    val_ds   = TensorDataset(*make_tensors(X_val,   y_val))
    test_ds  = TensorDataset(*make_tensors(X_test,  y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader