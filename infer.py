import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from auto_encoder import AutoEncoder

# 結果保存用のディレクトリを作成
os.makedirs('results', exist_ok=True)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# データセットの準備
transform = transforms.Compose([transforms.ToTensor()])

# テストデータセットをロード
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# データローダーの作成
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデルのロード
model = AutoEncoder().to(device)
model.load_state_dict(torch.load('autoencoder_model.pth', map_location=device))
model.eval()

# Fashion MNISTのクラス名
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]


# 1. 画像の再構成と可視化
def reconstruct_images() -> None:
    # サンプル画像を取得
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images.to(device)

    with torch.no_grad():
        reconstructed = model(images)

    # 10枚の画像を選択
    n = 10
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # 元の画像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"{class_names[labels[i]]}")
        plt.axis('off')

        # 再構成された画像
        ax = plt.subplot(2, n, i + 1 + n)  # 2行目のサブプロットを指定
        plt.imshow(reconstructed[i].cpu().squeeze().numpy(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/reconstructed_images.png')
    plt.close()
    print("Reconstructed images saved to results/reconstructed_images.png")


# 2. テストセットの埋め込みをプロット
def plot_embeddings() -> None:
    embeddings = []
    img_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            encoded = model.encode(images)
            embeddings.append(encoded.cpu().numpy())
            img_labels.append(labels.numpy())

    # リストを結合
    embeddings = np.vstack(embeddings)
    img_labels = np.concatenate(img_labels)

    # 散布図をプロット
    plt.figure(figsize=(10, 8))
    for i in range(10):
        idx = img_labels == i
        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            label=class_names[i],
            alpha=0.5,
            s=10,
        )

    plt.legend()
    plt.title("Latent Space Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig('results/latent_embeddings.png')
    plt.close()
    print("Latent embeddings plot saved to results/latent_embeddings.png")


# 3. 潜在空間からサンプリングして新しい画像を生成
def generate_from_latent() -> None:
    # 潜在空間にグリッドを作成
    n = 5  # グリッドのサイズ

    # 潜在空間の範囲を計算（既存の埋め込みから）
    all_embeddings = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            encoded = model.encode(images)
            all_embeddings.append(encoded.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    min_vals = np.min(all_embeddings, axis=0) - 0.5
    max_vals = np.max(all_embeddings, axis=0) + 0.5

    # グリッド上の点をサンプリング
    x = np.linspace(min_vals[0], max_vals[0], n)
    y = np.linspace(min_vals[1], max_vals[1], n)

    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # グリッド上の点からデコード
    with torch.no_grad():
        latent_samples = torch.tensor(grid_points, dtype=torch.float32).to(device)
        generated_images = model.decode(latent_samples)

    # 生成した画像をプロット
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(generated_images[i].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/generated_images_grid.png')
    plt.close()
    print("Generated images saved to results/generated_images_grid.png")

    # 潜在空間を可視化して生成点をマーク
    plt.figure(figsize=(10, 8))
    # 背景に全埋め込みをプロット
    plt.scatter(
        all_embeddings[:, 0], all_embeddings[:, 1], c='lightgray', alpha=0.3, s=5
    )
    # サンプリングした点をプロット
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c='red', s=50, marker='x')
    plt.title("Sampled Points in Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig('results/latent_samples.png')
    plt.close()
    print("Latent sampling plot saved to results/latent_samples.png")


# メイン処理を実行
if __name__ == "__main__":
    print("Starting inference process...")

    print("\n1. Reconstructing images...")
    reconstruct_images()

    print("\n2. Plotting embeddings...")
    plot_embeddings()

    print("\n3. Generating new images from latent space...")
    generate_from_latent()

    print("\nAll processes completed successfully!")
