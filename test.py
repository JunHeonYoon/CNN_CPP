import torch
import torch.nn as nn
import numpy as np
import torchsummary
import os
import time

def save_layer_params(layer, layer_path, layer_name):
    """주어진 레이어의 파라미터를 파일로 저장"""
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.Linear):
        os.makedirs(layer_path, exist_ok=True)
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight_file = os.path.join(layer_path, "weight.txt")
            np.savetxt(weight_file, layer.weight.data.cpu().numpy().flatten())
        if hasattr(layer, 'bias') and layer.bias is not None:
            bias_file = os.path.join(layer_path, "bias.txt")
            np.savetxt(bias_file, layer.bias.data.cpu().numpy())

def save_model_params(model, base_path="model"):
    """모델의 모든 레이어를 순회하며 파라미터를 저장"""
    conv_index = 0  # conv 블록 인덱스 추적
    conv_started = False  # Conv 블록이 시작되었는지 확인

    # Create base directory if it does not exist
    os.makedirs(base_path, exist_ok=True)

    for layer in model.children():
        layer_index = 0  # 레이어 인덱스를 추적하여 고유한 이름을 생성
        if isinstance(layer, nn.Sequential):
            conv_index += 1  # 새로운 Conv 블록 시작
            conv_path = os.path.join(base_path, f"conv{conv_index}")
            os.makedirs(conv_path, exist_ok=True)

            for sublayer in layer.children():
                layer_index += 1
                layer_name = sublayer.__class__.__name__.lower()
                layer_path = os.path.join(conv_path, f"{layer_index}_{layer_name}")
                
                # 파라미터 저장
                save_layer_params(sublayer, layer_path, layer_name)
        
        else:
            layer_index += 1
            layer_name = layer.__class__.__name__.lower()
            layer_path = os.path.join(base_path, f"{layer_index}_{layer_name}")

            # 파라미터 저장
            save_layer_params(layer, layer_path, layer_name)


class SimpleCNN(nn.Module):
    def __init__(self, dof):
        super(SimpleCNN, self).__init__()
        self.dof = dof

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear((self.dof+2) * 128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

dof = 7

# Create the model
model = SimpleCNN(dof)

# Specify device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Move the model to the device
model.to(device)
model.eval()

# Print model summary with correct input shape
# torchsummary.summary(model, (1, dof))

# Save input tensor
input = torch.randn(1, 1, dof).to(device)  # Ensure input is on the same device
np.savetxt('model/input.txt', input.cpu().numpy().flatten())
print("Input:\n", input.cpu().detach().numpy())

save_model_params(model, base_path="model")

# Forward pass
start = time.time()
output = model(input)
print("Duration: ", time.time() - start)
print("Output:\n", output.cpu().detach().numpy())
