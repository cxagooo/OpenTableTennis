import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(sequences):
    # 转换为 PyTorch 张量
    data_tensor = torch.tensor(sequences, dtype=torch.float32)

    # 数据归一化
    mean = data_tensor.mean(dim=(0, 1))
    std = data_tensor.std(dim=(0, 1))
    data_tensor = (data_tensor - mean) / std
    np.save('npy/mean.npy', mean.numpy())

    np.save('npy/std.npy', std.numpy())

    return data_tensor, mean, std
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM 层
        out, _ = self.lstm(x, (h0, c0))

        # 使用最后一个时间步的输出作为 LSTM 的输出
        out = self.fc(out[:, -1, :])
        return out
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 确保数据具有正确的形状
        sequence = self.sequences[idx]
        target = self.targets[idx]
        if sequence.dim() == 2:
            # 如果序列是一维的，则增加一个维度以表示序列长度
            sequence = sequence.unsqueeze(0)
        return sequence, target
def train_val_split(data, target, train_ratio=0.8):
    n_train = int(train_ratio * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    train_target, val_target = target[:n_train], target[n_train:]
    return train_data, train_target, val_data, val_target


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs):

    # 用于存储最佳模型的路径

    best_model_path = 'best_model.pth'

    # 初始化最佳验证损失为无穷大

    best_val_loss = float('inf')


    # 用于存储每轮的训练和验证损失

    train_losses = []

    val_losses = []


    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:

            # 确保输入数据具有正确的形状

            if inputs.dim() == 2:

                inputs = inputs.unsqueeze(1)


            # 确保标签具有正确的类型

            labels = labels.float()


            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        # 计算平均训练损失

        avg_train_loss = running_loss / len(train_loader)

        train_losses.append(avg_train_loss)


        # 验证

        model.eval()

        with torch.no_grad():

            val_loss = 0.0

            for inputs, labels in val_loader:

                # 确保输入数据具有正确的形状

                if inputs.dim() == 2:

                    inputs = inputs.unsqueeze(1)


                # 确保标签具有正确的类型

                labels = labels.float()


                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            val_losses.append(avg_val_loss)


        # 打印训练和验证损失

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


        # 如果当前验证损失是最低的，则保存模型

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(model.state_dict(), best_model_path)

            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")


    # 绘制损失曲线图

    plt.figure(figsize=(10, 5))

    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')

    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')

    plt.title('Training and Validation Loss Over Epochs')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.grid(True)


    # 显示并保存损失曲线图

    plt.savefig('loss_curve.png')

    plt.show()
# 假设 data.csv 是包含完整数据集的文件
data = pd.read_csv('data.csv')

# 提取特征和标签
sequences = data.iloc[:, 2:].values
targets = data.iloc[:, 2:].values  # 假设预测目标与输入特征相同

# 数据预处理
sequences, mean, std = preprocess_data(sequences)

# 划分数据集
train_sequences, train_targets, val_sequences, val_targets = train_val_split(sequences, targets)

# 创建数据集
train_dataset = SequenceDataset(train_sequences, train_targets)
val_dataset = SequenceDataset(val_sequences, val_targets)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义模型参数
input_size = 6  # 每个时间步的特征数量
hidden_size = 64
num_layers = 2
output_size = 6  # 输出特征数量
learning_rate = 0.001
epochs = 2000

# 初始化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练和验证
train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs)