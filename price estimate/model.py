import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, layer_dim, seq_dim):
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_dim = seq_dim

        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.3)
        # 全连接层输出连续值（股票价格）
        self.fc = nn.Linear(hidden_dim, 1)

        # 保存预测值和标签
        self.preds = []
        self.labels = []

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后时间步的输出
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)  # 更小的学习率
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred.squeeze(), y)  # 均方误差
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = F.mse_loss(y_pred.squeeze(), y)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.preds.append(y_pred.squeeze().cpu())
        self.labels.append(y.cpu())

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        test_loss = F.mse_loss(y_pred.squeeze(), y)
        self.log("test_loss", test_loss, prog_bar=True, sync_dist=True)
        self.preds.append(y_pred.squeeze().cpu())
        self.labels.append(y.cpu())

    def on_validation_epoch_end(self):
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)

        mae = mean_absolute_error(labels.numpy(), preds.numpy())
        mse = mean_squared_error(labels.numpy(), preds.numpy())
        rmse = mse ** 0.5
        r2 = r2_score(labels.numpy(), preds.numpy())

        self.log("Val MAE", mae, prog_bar=True)
        self.log("Val RMSE", rmse, prog_bar=True)
        self.log("Val R²", r2, prog_bar=True)

        # 清空
        self.preds.clear()
        self.labels.clear()

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)

        mae = mean_absolute_error(labels.numpy(), preds.numpy())
        mse = mean_squared_error(labels.numpy(), preds.numpy())
        rmse = mse ** 0.5
        r2 = r2_score(labels.numpy(), preds.numpy())

        self.log("Test MAE", mae, prog_bar=True)
        self.log("Test RMSE", rmse, prog_bar=True)
        self.log("Test R²", r2, prog_bar=True)

        # 清空
        self.preds.clear()
        self.labels.clear()

    def predict_step(self, batch, batch_idx):
        x, _ = batch  # 预测阶段不需要标签
        x = x.view(x.size(0), self.seq_dim, -1)
        pred = self.forward(x)
        return pred  # 返回预测的股票价格


