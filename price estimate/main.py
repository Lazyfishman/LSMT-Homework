import os
import sys
import pandas as pd
import lightning as L
from stock_dataset import StockDataset
from model import LSTMModel
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# 强制设置环境编码为 UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

def main():
    seq_length = 50
    batch_size = 64

    # 加载数据并进行训练验证集拆分
    data = pd.read_csv("Tesla_Stock_Data.csv")
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = StockDataset(data=train_data, seq_length=seq_length, target_col="Close")
    val_dataset = StockDataset(data=val_data, seq_length=seq_length, target_col="Close")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 获取输入维度
    input_dim = train_dataset.features.shape[1]

    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="lstm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # 初始化模型
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=128,
        layer_dim=2,
        seq_dim=seq_length
    )

    # 初始化 Trainer
    trainer = L.Trainer(
        max_epochs=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        default_root_dir="checkpoints"
    )

    # 模型训练
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
