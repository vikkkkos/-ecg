import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class ECG_Lense(nn.Module):
    def __init__(self, input_shape, classes):
        super(ECG_Lense, self).__init__()

        self.conv = self.make_layers(input_shape)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=4 * 128, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1024, out_features=classes)
        )

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': [], 'lr': []
        }

    def make_layers(self, input_shape):
        conv_blocks = []
        in_channels = input_shape[1]

        for i in range(4):
            out_channels = (i + 1) * 128
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(0.4)
            )
            conv_blocks.append(block)
            in_channels = out_channels

        return nn.Sequential(*conv_blocks)

    def forward(self, x):
        out = self.conv(x)
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, patience, min_delta, verbose):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        best_val_loss = float("inf")
        es_counter = 0
        best_weights = None

        X_train_tensor = torch.FloatTensor(X_train).permute(0, 2, 1)
        y_train_tensor = torch.LongTensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).permute(0, 2, 1)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        print(f"Training on {device}")
        print("-" * 60)

        for epoch in range(epochs):

            self.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                probs = torch.softmax(output, dim=1)
                train_preds.extend(probs.cpu().detach().numpy())
                train_targets.extend(target.cpu().numpy())

            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_targets, np.argmax(train_preds, axis=1))
            train_auc = roc_auc_score(train_targets, train_preds, multi_class='ovr')

            val_loss = 0.0
            val_preds = []
            val_targets = []

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = self(data)
                        loss = criterion(output, target)

                        val_loss += loss.item()

                        probs = torch.softmax(output, dim=1)
                        val_preds.extend(probs.cpu().numpy())
                        val_targets.extend(target.cpu().numpy())

                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_targets, np.argmax(val_preds, axis=1))
                val_auc = roc_auc_score(val_targets, val_preds, multi_class='ovr')

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])

            scheduler.step(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"loss {train_loss:.4f} | acc {train_acc:.4f} | "
                    f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
                    f"val_auc {val_auc:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}"
                )

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                es_counter = 0
                best_weights = self.state_dict()
            else:
                es_counter += 1
                if es_counter >= patience:
                    print(f"\n Early Stopping на эпохе {epoch+1}")
                    break

        if best_weights is not None:
            self.load_state_dict(best_weights)

    def predict(self, X, batch_size):
        device = next(self.parameters()).device
        X_tensor = torch.FloatTensor(X).permute(0, 2, 1)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.eval()
        preds = []

        with torch.no_grad():
            for (data,) in dataloader:
                data = data.to(device)
                output = self(data)
                probs = torch.softmax(output, dim=1)
                preds.extend(probs.cpu().numpy())

        return np.array(preds)