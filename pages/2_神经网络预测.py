import streamlit as st
import streamlit_ace as st_ace
from io import StringIO
import contextlib
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math



# 定义神经网络结构
class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(64, 128)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(128, 64)  # 第二个隐藏层到第三个隐藏层
        self.fc4 = nn.Linear(64, 2)  # 第三个隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(ComplexLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)

        # 添加Dropout层
        self.dropout = nn.Dropout(dropout_prob)

        # 定义额外的全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)  # 批量归一化层
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # 前向传播LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 只需要LSTM最后一层的输出
        out = out[:, -1, :]
        out = self.dropout(out)

        # 通过全连接层
        out = self.fc1(out)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out
# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, output_dim, num_layers, dropout_prob):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout_prob)

        # 由于3是质数，我们可以选择将输入特征投影到更高的维度
        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=8,
                                                 dim_feedforward=hidden_dim, dropout=dropout_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.embedding(x)  # 将输入特征投影到更高的维度
        x = self.dropout(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects (Seq Len, Batch, Features)
        out = self.transformer_encoder(x)
        out = self.fc_out(out[-1, :, :])  # 取最后一个时间步
        return out


def evaluate_model(model, test_loader):
    model.eval()  # 将模型设置为评估模式
    predictions = []
    actuals = []
    with torch.no_grad():  # 不计算梯度
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 移动数据到GPU
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # 反标准化以比较实际值
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)

    # 计算均方根误差
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)

    # 计算平均绝对误差
    mae = mean_absolute_error(actuals, predictions)

    return rmse, mae
# 加载和准备数据
data = pd.read_csv('C://Users//Administrator//Desktop//ChatTTS-main//pages//earthquake.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
data['Timestamp'] = data.apply(lambda row: pd.Timestamp(f"{row['Date']} {row['Time']}"), axis=1)
data['Timestamp'] = data['Timestamp'].view('int64') // 10**9 #Unix时间戳
features = data[['Timestamp', 'Longitude', 'Latitude']]
targets = data[['Depth', 'Magnitude']]
X_np=features.values
y_np=targets.values
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X_np)
scaler_y.fit(y_np)




device = torch.device("cpu")
# 初始化网络
model_BP = ComplexNet()
model_BP.to(device)
# 初始化模型
input_dim = 3 # 维度、经度、时间
hidden_dim = 64
num_layers = 2
output_dim = 2 # 震深和震级
dropout_prob = 0.5

model_lstm = ComplexLSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)

# 将模型移到GPU
model_lstm = model_lstm.to(device)
# 初始化模型参数
input_dim = 3  # 时间、经度、纬度
seq_len = 20   # 序列长度
hidden_dim = 64  # 隐藏层的维度
num_layers = 2  # Transformer层的数量
output_dim = 2  # 震深和震级
dropout_prob = 0.5  # Dropout概率

# 创建模型
model_transformer = TransformerModel(input_dim, seq_len, hidden_dim, output_dim, num_layers, dropout_prob)

# 将模型移到GPU
model_transformer = model_transformer.to(device)
# 加载最佳模型
model_BP.load_state_dict(torch.load('C://Users//Administrator//Desktop//ChatTTS-main//minloss_BP_model.pth', map_location=torch.device('cpu')))
model_BP.to(device)  # 确保模型在正确的设备上
# 加载最佳模型
model_lstm.load_state_dict(torch.load('C://Users//Administrator//Desktop//ChatTTS-main//best_lstm_model.pth', map_location=torch.device('cpu')))
model_lstm.to(device)  # 确保模型在正确的设备上
# 加载最佳模型
model_transformer.load_state_dict(torch.load('C://Users//Administrator//Desktop//ChatTTS-main//best_transformer_model.pth', map_location=torch.device('cpu')))
model_transformer.to(device)  # 确保模型在正确的设备上


def BP_forward(time, Latitude, Longitude):
    '''
    time:1965-01-02 13:44:18

    '''
    model=model_BP
    # 切换到评估模式
    model.eval()
    time = pd.to_datetime(time)
    time = time.timestamp()  # Unix时间戳
    features = np.array([[Latitude, Longitude, time]])
    X = torch.tensor(features, dtype=torch.float32)
    # 不计算梯度
    with torch.no_grad():
        X = X.to(device)
        output = model(X)
    return output


def lstm_forward(time, Latitude, Longitude):
    model=model_lstm
    model.eval() # 将模型设置为评估模式
    predictions = []
    time = pd.to_datetime(time)
    time=time.timestamp() #Unix时间戳
    features=np.array([[Latitude, Longitude, time]])
    X = scaler_x.transform(features)
    X = torch.tensor(X, dtype=torch.float32)
    X = X.reshape(1,1,3)
    with torch.no_grad(): # 不计算梯度
        X = X.to(device)
        outputs = model(X)
        predictions.append(outputs.cpu().numpy())

    # 反标准化以比较实际值

    predictions = np.vstack(predictions)
    predictions = scaler_y.inverse_transform(predictions)
    return predictions
def transformer_forward(time, Latitude, Longitude):
    model=model_transformer
    model.eval() # 将模型设置为评估模式
    predictions = []
    time = pd.to_datetime(time)
    time=time.timestamp() #Unix时间戳
    features=np.array([[Latitude, Longitude, time]])
    X = scaler_x.transform(features)
    X = torch.tensor(X, dtype=torch.float32)
    X = X.reshape(1,1,3)
    with torch.no_grad(): # 不计算梯度
        X = X.to(device)
        outputs = model(X)
        predictions.append(outputs.cpu().numpy())

    # 反标准化以比较实际值

    predictions = np.vstack(predictions)
    predictions = scaler_y.inverse_transform(predictions)
    return predictions
# 定义一段默认代码
DEFAULT_CODE_BP = """
# 加载和准备数据
data = pd.read_csv('C://Users//Administrator//Desktop//ChatTTS-main//pages//earthquake.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
data['Timestamp'] = data.apply(lambda row: pd.Timestamp(f"{row['Date']} {row['Time']}"), axis=1)
data['Timestamp'] = data['Timestamp'].view('int64') // 10**9 #Unix时间戳
features = data[['Latitude', 'Longitude', 'Timestamp']]
targets = data[['Depth', 'Magnitude']]
X = torch.tensor(features.values, dtype=torch.float32)
Y = torch.tensor(targets.values, dtype=torch.float32)
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# 计算分割尺寸
test_size = int(len(dataset) * 0.2)  # 20%作为测试集
train_size = len(dataset) - test_size  # 剩余作为训练集

# 随机分割数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 你可以将这些数据集进一步封装成DataLoader，便于批处理和迭代
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 从test_dataset中提取X_test和Y_test
# 注意：random_split返回的子集类型是Subset，我们需要通过.dataset属性访问原始数据
X_test = torch.stack([data[0] for data in test_dataset])
Y_test = torch.stack([data[1] for data in test_dataset])

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 如果有多个GPU，可以更改索引从 0 到 1, 2, ...
    print("Training on GPU...")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Training on CPU...")


# 初始化网络
model = ComplexNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
# 训练网络
min_loss=10000
epochs = 200
for epoch in range(epochs):
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)  # 移动到GPU

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y)
        loss.backward()
        optimizer.step()
    

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    if min_loss>loss.item():
        torch.save(model.state_dict(), 'minloss_BP_model.pth')
        min_loss=loss.item()
predictions = predictions.cpu()
# 保存模型
torch.save(model.state_dict(), 'lastloss_BP_model.pth')
"""
DEFAULT_CODE_LSTM = """
# 加载和准备数据
data = pd.read_csv('C://Users//Administrator//Desktop//ChatTTS-main//pages//earthquake.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
data['Timestamp'] = data.apply(lambda row: pd.Timestamp(f"{row['Date']} {row['Time']}"), axis=1)
data['Timestamp'] = data['Timestamp'].view('int64') // 10**9 #Unix时间戳
features = data[['Timestamp', 'Longitude', 'Latitude']]
targets = data[['Depth', 'Magnitude']]

X_np=features.values
y_np=targets.values
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X_np)
scaler_y.fit(y_np)

# 使用相同的参数对训练数据和测试数据进行归一化
X_normalized = scaler_x.transform(X_np)
y_normalized = scaler_y.transform(y_np)
def create_sliding_windows(X, Y, window_size):
    X_windows, Y_windows = [], []
    for i in range(len(X) - window_size):
        # 从X中提取窗口
        X_window = X[i:i+window_size]
        # 从Y中提取紧接着窗口的下一个值作为标签
        Y_window = Y[i+window_size]
        X_windows.append(X_window)
        Y_windows.append(Y_window)
    return np.array(X_windows), np.array(Y_windows)

window_size = 10
X_windows, Y_windows = create_sliding_windows(X_normalized, y_normalized, window_size)
X = torch.tensor(X_windows, dtype=torch.float32)
Y = torch.tensor(Y_windows, dtype=torch.float32)
# 确定测试集的大小
test_size = 0.2

# 计算测试集应有的数据点数量
num_data_points = len(X)
num_test_points = int(num_data_points * test_size)

# 计算测试集和训练集的分割点
split_point = num_data_points - num_test_points

# 按顺序划分数据为训练集和测试集
X_train = X[:split_point]
X_test = X[split_point:]
y_train = Y[:split_point]
y_test = Y[split_point:]
# 创建DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
batch_size=64
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
# 初始化模型
input_dim = 3  # 维度、经度、时间
hidden_dim = 64
num_layers = 2
output_dim = 2  # 震深和震级
dropout_prob = 0.5

model = ComplexLSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移到GPU
model = model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练轮数
num_epochs = 200

# 用于保存最佳模型的逻辑
best_loss = np.inf
best_model_path = 'best_model.pth'
# 训练模型
for epoch in range(num_epochs):
    model.train() # 确保模型处于训练模式
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device) # 移动数据到GPU
        optimizer.zero_grad() # 清除过往梯度
        
        outputs = model(inputs) # 前向传播
        
        loss = criterion(outputs, targets) # 计算损失
        loss.backward() # 后向传播，计算梯度
        optimizer.step() # 更新权重
    
    # 打印训练进度

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 保存最佳模型
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), best_model_path)
"""
DEFAULT_CODE_Transformer = """
# 加载和准备数据
data = pd.read_csv('C://Users//Administrator//Desktop//ChatTTS-main//pages//earthquake.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
data['Timestamp'] = data.apply(lambda row: pd.Timestamp(f"{row['Date']} {row['Time']}"), axis=1)
data['Timestamp'] = data['Timestamp'].view('int64') // 10**9 #Unix时间戳
features = data[['Timestamp', 'Longitude', 'Latitude']]
targets = data[['Depth', 'Magnitude']]
X_np=features.values
y_np=targets.values
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X_np)
scaler_y.fit(y_np)

# 使用相同的参数对训练数据和测试数据进行归一化
X_normalized = scaler_x.transform(X_np)
y_normalized = scaler_y.transform(y_np)
def create_sliding_windows(X, Y, window_size):
    X_windows, Y_windows = [], []
    for i in range(len(X) - window_size):
        # 从X中提取窗口
        X_window = X[i:i+window_size]
        # 从Y中提取紧接着窗口的下一个值作为标签
        Y_window = Y[i+window_size]
        X_windows.append(X_window)
        Y_windows.append(Y_window)
    return np.array(X_windows), np.array(Y_windows)

window_size = 20
X_windows, Y_windows = create_sliding_windows(X_normalized, y_normalized, window_size)
X = torch.tensor(X_windows, dtype=torch.float32)
Y = torch.tensor(Y_windows, dtype=torch.float32)
# 确定测试集的大小
test_size = 0.2

# 计算测试集应有的数据点数量
num_data_points = len(X)
num_test_points = int(num_data_points * test_size)

# 计算测试集和训练集的分割点
split_point = num_data_points - num_test_points

# 按顺序划分数据为训练集和测试集
X_train = X[:split_point]
X_test = X[split_point:]
y_train = Y[:split_point]
y_test = Y[split_point:]
# 创建DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
batch_size=64
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
# 初始化模型参数
input_dim = 3  # 时间、经度、纬度
seq_len = 20   # 序列长度
hidden_dim = 64  # 隐藏层的维度
num_layers = 2  # Transformer层的数量
output_dim = 2  # 震深和震级
dropout_prob = 0.5  # Dropout概率

# 创建模型
model = TransformerModel(input_dim, seq_len, hidden_dim, output_dim, num_layers, dropout_prob)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移到GPU
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 因为是回归问题
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练轮数
num_epochs = 200
best_loss = np.inf
best_model_path = 'best_transformer_model.pth'

for epoch in range(num_epochs):
    model.train()  # 确保模型处于训练模式
    epoch_loss = 0.0  # 用于累积整个epoch的损失

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # 移动数据到GPU
        
        optimizer.zero_grad()  # 清除过往梯度
        
        outputs = model(inputs)  # 前向传播
        
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 更新权重

        epoch_loss += loss.item() * inputs.size(0)  # 累积损失

    # 计算整个epoch的平均损失
    epoch_loss /= len(train_loader.dataset)
    
    # 打印训练进度

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # 保存最佳模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
torch.save(model.state_dict(), 'last_transformer_model.pth')


"""
def execute_code(code):
    # 创建一个字符串流来捕获执行输出
    output = StringIO()
    with contextlib.redirect_stdout(output):
        with contextlib.redirect_stderr(output):
            try:
                # 执行用户提供的代码
                exec(code)
            except Exception as e:
                # 如果代码执行中出现错误，打印错误信息
                print(f"Error: {e}")
    # 获取输出并返回
    return output.getvalue()

def main():
    st.title("神经网络预测")


    # 用户选择模型
    genre = st.radio(
        "Select prediction model:",
        ["⚡ BP", "🔁 LSTM", "🤖 Transformer"],
        captions=["BP Neural Network Model", "LSTM Neural Network Model", "Transformer Neural Network Model"]
    )

    # 获取用户输入的参数
    time = st.text_input("Enter Time (YYYY-MM-DD HH:MM:SS)", "1983-01-02 13:44:18")

    Latitude = st.number_input("Enter Latitude", value=19.246, format="%.2f")
    Longitude = st.number_input("Enter Longitude", value=145.616, format="%.2f")
    # 检查是否已输入所有参数，并定义按钮触发预测
    depth=None
    magnitude=None
    a=None
    if st.button("Predict"):
        # 根据选择的模型调用不同的函数
        if genre == "⚡ BP":
            a = BP_forward(time, Latitude, Longitude)
            depth = a[0][0]
            magnitude = a[0][1]

        elif genre == "🔁 LSTM":
            a = lstm_forward(time, Latitude, Longitude)
            depth = a[0][0]
            magnitude = a[0][1]
        elif genre == "🤖 Transformer":
            a = transformer_forward(time, Latitude, Longitude)
            depth = a[0][0]
            magnitude = a[0][1]
        # 显示预测结果
        st.write(f"Predicted Depth: {depth} km")
        st.write(f"Predicted Magnitude: {magnitude}")
    else:
        st.write("Enter all parameters to get the prediction.")

    st.markdown('***BP人工神经网络训练***')
    # 使用 streamlit-ace 组件让用户可以输入代码，同时展示默认代码
    code_bp = st_ace.st_ace(language='python', theme='monokai', value=DEFAULT_CODE_BP)
    if st.button("Execute Code BP"):
        # 当用户点击执行时，调用 execute_code 函数
        output = execute_code(code_bp)
        st.text_area("Output", output, height=300)
    st.markdown('***LSTM***')
    # 使用 streamlit-ace 组件让用户可以输入代码，同时展示默认代码
    code_lstm = st_ace.st_ace(language='python', theme='monokai', value=DEFAULT_CODE_LSTM)
    if st.button("Execute Code LSTM"):
        # 当用户点击执行时，调用 execute_code 函数
        output = execute_code(code_lstm)
        st.text_area("Output", output, height=300)
    st.markdown('***Transformer***')
    # 使用 streamlit-ace 组件让用户可以输入代码，同时展示默认代码
    code_Transformer = st_ace.st_ace(language='python', theme='monokai', value=DEFAULT_CODE_Transformer)
    if st.button("Execute Code Transformer"):
        # 当用户点击执行时，调用 execute_code 函数
        output = execute_code(code_Transformer)
        st.text_area("Output", output, height=300)

if __name__ == "__main__":
    main()
