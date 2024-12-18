import os
import sys
import pandas as pd
import lightning as L
from stock_dataset import StockDataset
from model import LSTMModel
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import csv
from lightning.pytorch.loggers import CSVLogger


# 强制设置环境编码为 UTF-8，确保运行环境兼容
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# 自定义 CSVLogger，重写保存方法，使用 UTF-8 编码
class UTF8CSVLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self):
        # 动态获取所有字段名
        metrics = self.experiment.metrics
        if len(metrics) > 0:
            # 提取所有键并去重
            metrics_keys = set(key for metric in metrics for key in metric.keys())
        else:
            metrics_keys = set()

        # 确保字段名是有序的
        metrics_keys = sorted(metrics_keys)

        # 确保使用 UTF-8 编码保存日志
        with open(self.experiment.metrics_file_path, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(metrics)


def main():
    seq_length = 50
    batch_size = 64

    # 加载数据并进行训练验证集拆分
    data = pd.read_csv("Tesla_Stock_Data.csv")
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # 初始化数据集
    train_dataset = StockDataset(data=train_data, seq_length=seq_length, target_col="Close")
    val_dataset = StockDataset(data=val_data, seq_length=seq_length, target_col="Close")

    # 初始化数据加载器
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

    # 初始化自定义 CSVLogger
    csv_logger = UTF8CSVLogger("logs", name="my_model")

    # 初始化模型
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=128,
        layer_dim=2,
        seq_dim=seq_length
    )

    # 初始化 Trainer
    trainer = L.Trainer(
        max_epochs=30,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        default_root_dir="checkpoints",
        logger=csv_logger  # 使用自定义日志器
    )

    # 模型训练
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
