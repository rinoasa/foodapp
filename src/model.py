import os
import torch
from torchvision import datasets, transforms
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import Food101
import copy
from torch.optim import lr_scheduler
import time

# 前処理とデータ拡張を定義する関数
def get_transforms():
    """
    データ拡張と前処理の設定を行います。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 画像を224x224にリサイズ
        transforms.RandomRotation(5),     # 画像を5度までランダムに回転
        transforms.RandomHorizontalFlip(), # ランダムに左右反転
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 色の変化
        transforms.ToTensor(),  # Tensor形式に変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNetの平均と標準偏差で正規化
                             std=[0.229, 0.224, 0.225])
    ])

# Food101データセットの読み込み
train_data = Food101(root='data', split='train', download=True, transform=get_transforms())
test_data = Food101(root='data', split='test', download=True, transform=get_transforms())

# 訓練データの分割（80%訓練、20%検証）
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# DataLoaderの設定（バッチサイズを調整）
batch_size = 64  # ここでバッチサイズを設定（64, 128, 256など）
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 訓練データのDataLoader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 検証データのDataLoader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  # テストデータのDataLoader

# クラス名の取得
classes = train_data.classes
print(f"Classes: {classes}")

# 事前学習済みのResNet50モデルをロード
model = models.resnet50(pretrained=True)

# 最後の層とlayer4のパラメータを訓練可能に設定
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:  # 最後の層とlayer4のパラメータを訓練可能に
        param.requires_grad = True
    else:
        param.requires_grad = False  # 他の層は訓練しない

# 出力層をFood101用にカスタマイズ（ドロップアウト率を調整）
num_ftrs = model.fc.in_features  # 入力特徴量数を取得
model.fc = nn.Sequential(
    nn.Dropout(0.3),  # ドロップアウト率を0.3に設定
    nn.Linear(num_ftrs, len(classes))  # 出力層のクラス数を設定
)

# モデルをデバイスに移動（GPUが使用可能ならGPUへ）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()  # クロスエントロピー損失関数を使用

# オプティマイザ（Adamに変更）
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=0.001)  # 学習率を0.001に設定

# 学習率スケジューラの設定（ReduceLROnPlateauを使用）
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# 損失と精度のグラフを作成する関数
def create_loss_plot(train_losses, val_losses):
    """
    損失をプロットする関数
    """
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')  # 訓練損失をプロット
    ax.plot(val_losses, label='Validation Loss')  # 検証損失をプロット
    ax.set_xlabel('Epoch')  # x軸のラベル
    ax.set_ylabel('Loss')  # y軸のラベル
    ax.set_title('Loss over Epochs')  # タイトル
    ax.legend()  # 凡例を表示
    return fig  # プロットを返す

def create_accuracy_plot(train_accuracies, val_accuracies):
    """
    精度をプロットする関数
    """
    fig, ax = plt.subplots()
    ax.plot(train_accuracies, label='Training Accuracy')  # 訓練精度をプロット
    ax.plot(val_accuracies, label='Validation Accuracy')  # 検証精度をプロット
    ax.set_xlabel('Epoch')  # x軸のラベル
    ax.set_ylabel('Accuracy')  # y軸のラベル
    ax.set_title('Accuracy over Epochs')  # タイトル
    ax.legend()  # 凡例を表示
    return fig  # プロットを返す

# トレーニング関数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30):
    """
    モデルを訓練する関数
    """
    model.to(device)  # モデルをデバイスに移動
    best_model_wts = copy.deepcopy(model.state_dict())  # 最良のモデルの重みを保存
    best_acc = 0.0  # 最良の精度を初期化

    train_losses = []  # 訓練損失を保存するリスト
    val_losses = []    # 検証損失を保存するリスト
    train_accuracies = []  # 訓練精度を保存するリスト
    val_accuracies = []  # 検証精度を保存するリスト

    start_time = time.time()  # 訓練開始時間を記録

    for epoch in range(num_epochs):  # 各エポックに対して
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 訓練フェーズ
        model.train()  # モデルを訓練モードに設定
        running_loss = 0.0  # 累積損失の初期化
        running_corrects = 0  # 正解数の初期化

        for inputs, labels in tqdm(train_loader):  # 各バッチに対して
            inputs = inputs.to(device)  # 入力データをデバイスに移動
            labels = labels.to(device)  # ラベルをデバイスに移動

            optimizer.zero_grad()  # 勾配をゼロに初期化

            outputs = model(inputs)  # モデルの出力を取得
            _, preds = torch.max(outputs, 1)  # 予測を取得
            loss = criterion(outputs, labels)  # 損失を計算

            loss.backward()  # 勾配を計算
            optimizer.step()  # パラメータを更新

            running_loss += loss.item() * inputs.size(0)  # 累積損失を更新
            running_corrects += torch.sum(preds == labels.data)  # 正解数を更新

        epoch_loss = running_loss / len(train_loader.dataset)  # エポックの平均損失を計算
        epoch_acc = running_corrects.double() / len(train_loader.dataset)  # エポックの精度を計算
        train_losses.append(epoch_loss)  # 訓練損失を保存
        train_accuracies.append(epoch_acc.item())  # 訓練精度を保存

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')  # 訓練結果を表示

        # 検証フェーズ
        model.eval()  # モデルを評価モードに設定
        val_running_loss = 0.0  # 検証累積損失の初期化
        val_running_corrects = 0  # 検証正解数の初期化

        with torch.no_grad():  # 勾配を計算しない
            for inputs, labels in val_loader:  # 各バッチに対して
                inputs = inputs.to(device)  # 入力データをデバイスに移動
                labels = labels.to(device)  # ラベルをデバイスに移動

                outputs = model(inputs)  # モデルの出力を取得
                _, preds = torch.max(outputs, 1)  # 予測を取得
                loss = criterion(outputs, labels)  # 損失を計算

                val_running_loss += loss.item() * inputs.size(0)  # 検証損失を更新
                val_running_corrects += torch.sum(preds == labels.data)  # 正解数を更新

        val_loss = val_running_loss / len(val_loader.dataset)  # 検証エポックの平均損失を計算
        val_acc = val_running_corrects.double() / len(val_loader.dataset)  # 検証エポックの精度を計算
        val_losses.append(val_loss)  # 検証損失を保存
        val_accuracies.append(val_acc.item())  # 検証精度を保存

        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')  # 検証結果を表示

        # ベストモデルの保存
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())  # 最良のモデルの重みを保存

        # 学習率の更新
        scheduler.step(val_loss)

    elapsed_time = time.time() - start_time  # 実行時間を計測
    print(f'Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')  # 終了時間を表示

    # ログファイルに記録
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f'Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s\n')

    print(f'Best Validation Accuracy: {best_acc:.4f}')  # ベスト精度を表示

    # ベストモデルのロード
    model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses, train_accuracies, val_accuracies  # 結果を返す

# モデルの訓練
model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30)

# 損失と精度のグラフを作成
loss_fig = create_loss_plot(train_losses, val_losses)
accuracy_fig = create_accuracy_plot(train_accuracies, val_accuracies)

# 最後に画像として保存
loss_fig.savefig('loss_plot.png')
accuracy_fig.savefig('accuracy_plot.png')

# モデルの評価
def evaluate_model(model, test_loader, device):
    """
    テストデータでモデルを評価する関数
    """
    model.eval()  # モデルを評価モードに設定
    correct = 0  # 正解数の初期化
    total = 0  # 総サンプル数の初期化
    test_loss = 0.0  # テスト損失の初期化
    criterion = nn.CrossEntropyLoss()  # 損失関数の定義

    with torch.no_grad():  # 勾配を計算しない
        for inputs, labels in tqdm(test_loader):  # 各バッチに対して
            inputs, labels = inputs.to(device), labels.to(device)  # デバイスに移動

            outputs = model(inputs)  # モデルの出力を取得
            loss = criterion(outputs, labels)  # 損失を計算
            test_loss += loss.item() * inputs.size(0)  # テスト損失を更新

            _, predicted = torch.max(outputs, 1)  # 予測を取得
            total += labels.size(0)  # 総サンプル数を更新
            correct += (predicted == labels).sum().item()  # 正解数を更新

    accuracy = 100 * correct / total  # 精度を計算
    test_loss = test_loss / total  # テストの平均損失を計算
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')  # テスト結果を表示

    # ログファイルにテスト結果を記録
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n')

# テストデータでの評価
evaluate_model(model, test_loader, device)

# モデルの保存
torch.save(model.state_dict(), 'best_food101_model.pth')  # モデルの重みを保存
