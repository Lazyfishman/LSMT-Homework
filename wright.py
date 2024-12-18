from model import LSTMModel
import lightning as L
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from model import LSTMModel

# 数据加载
transform = transforms.ToTensor()
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, num_workers=2)

# 加载模型权重文件
checkpoint_path = "checkpoints/lstm-epoch=06-val_loss=0.07.ckpt"  # 修改为你的权重文件路径
model = LSTMModel.load_from_checkpoint(checkpoint_path)

torch.save(model.state_dict(), "lstm_model_weights.pth")
print("权重文件已保存为 lstm_model_weights.pth")

# 测试模型
trainer = L.Trainer()
trainer.test(model=model, dataloaders=test_loader)