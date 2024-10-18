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

# データ拡張と前処理の設定（回転を削除）
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # スケールを調整
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNetの平均と標準偏差
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Food101データセットの読み込み
train_data = Food101(root='data', split='train', download=True, transform=transform_train)
test_data = Food101(root='data', split='test', download=True, transform=transform_test)

# 訓練データの分割（80%訓練、20%検証）
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# DataLoaderの設定（バッチサイズを調整）
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# クラス名の取得
classes = train_data.classes
print(f"Classes: {classes}")

# 事前学習済みのResNet50モデルをロード
model = models.resnet50(pretrained=True)

# layer4とfc層のパラメータを訓練可能に設定
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 出力層をFood101用にカスタマイズ（ドロップアウト率を調整）
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),  # ドロップアウト率を0.3に設定
    nn.Linear(num_ftrs, len(classes))
)

# モデルをデバイスに移動
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# オプティマイザ（SGDに変更し、学習率と重み減衰を調整）
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=0.001, momentum=0.9, weight_decay=5e-4)

# 学習率スケジューラの設定（ReduceLROnPlateauを使用）
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# 損失と精度のグラフを変数に格納し、後で保存する関数
def create_loss_plot(train_losses, val_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss over Epochs')
    ax.legend()
    return fig

def create_accuracy_plot(train_accuracies, val_accuracies):
    fig, ax = plt.subplots()
    ax.plot(train_accuracies, label='Training Accuracy')
    ax.plot(val_accuracies, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy over Epochs')
    ax.legend()
    return fig

# トレーニング関数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()  # 訓練開始時間の記録

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 訓練フェーズ
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 検証フェーズ
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # ベストモデルの保存
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # 学習率の更新
        scheduler.step(val_loss)

    elapsed_time = time.time() - start_time  # 実行時間の計測
    print(f'Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')

    # ログファイルに記録
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f'Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s\n')

    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # ベストモデルのロード
    model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses, train_accuracies, val_accuracies

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
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = test_loss / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    # ログファイルにテスト結果を記録
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%\n')

# テストデータでの評価
evaluate_model(model, test_loader, device)

# モデルの保存
torch.save(model.state_dict(), 'best_food101_model.pth')