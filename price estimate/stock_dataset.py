import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, csv_path=None, data=None, seq_length=50, target_col="Close"):
        self.seq_length = seq_length

        # 允许从 CSV 文件加载或者直接从 DataFrame 传入
        if csv_path:
            self.data = pd.read_csv(csv_path)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("必须提供 csv_path 或 data 参数！")

        # 检查并删除非数值型列（如日期列）
        if "Date" in self.data.columns:
            self.data.drop(columns=["Date"], inplace=True, errors="ignore")

        # 检查是否有目标列
        if target_col not in self.data.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在于数据中！")

        # 检查并处理缺失值
        print("检查数据中的 NaN:")
        print(self.data.isnull().sum())  # 打印每列中缺失值的数量
        self.data.fillna(0, inplace=True)  # 用 0 填充所有 NaN

        # 确保所有列为数值型
        self.data = self.data.apply(pd.to_numeric, errors="coerce")
        print("检查是否有非数值列:")
        print(self.data.dtypes)

        # 初始化归一化器并保存
        self.scaler = MinMaxScaler()
        self.data[self.data.columns] = self.scaler.fit_transform(self.data)

        # 提取特征和目标列
        self.features = self.data.drop(columns=[target_col]).values.astype(np.float32)
        self.targets = self.data[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def inverse_transform(self, data):
        """反归一化，便于预测值还原"""
        return self.scaler.inverse_transform(data)
