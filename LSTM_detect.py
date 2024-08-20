import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, output_size=6)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()  # 设置模型为评估模式

# 加载归一化参数
mean = np.load('npy/mean.npy')
std = np.load('npy/std.npy')

# 加载新数据
new_data = ...  # 新的数据，例如从文件读取
new_data = np.array(new_data, dtype=np.float32)

# 使用训练数据的均值和标准差对新数据进行归一化
new_data = (new_data - mean) / std

# 将数据转换为 PyTorch 张量
new_data_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)

# 创建数据集
dataset = TensorDataset(new_data_tensor)

# 创建数据加载器
batch_size = 32  # 根据您的需求设置
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 预测
predictions = []
with torch.no_grad():
    for inputs in dataloader:
        inputs = inputs[0]  # 假设TensorDataset中的第一个元素是输入
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

# 打印预测结果
print("Predictions:", predictions)