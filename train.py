import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from auto_encoder import AutoEncoder

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ハイパーパラメータ
batch_size = 128
learning_rate = 1e-3
epochs = 5

# データセットの準備
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # [0, 255] -> [0, 1]の範囲にスケーリング
        transforms.Pad(2),  # 28x28 -> 32x32にパディング
    ]
)

# Fashion MNISTデータセットをダウンロード
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# モデルの初期化
model = AutoEncoder().to(device)


# RMSEロス関数の定義
def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


# オプティマイザの設定
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練ループ
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        # データをデバイスに移動
        data = data.to(device)

        # オプティマイザのリセット
        optimizer.zero_grad()

        # 順伝播
        outputs = model(data)

        # 損失の計算
        loss = rmse_loss(outputs, data)

        # 逆伝播
        loss.backward()

        # パラメータの更新
        optimizer.step()

        # 損失を累積
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    # エポックごとの平均損失
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {epoch_loss:.4f}")

# モデルの保存
torch.save(model.state_dict(), 'autoencoder_model.pth')
print("Model saved to autoencoder_model.pth")
