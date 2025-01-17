from fastapi import FastAPI, File, UploadFile
import os
import torch
from torchvision import models, transforms
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from src.database import get_food_calories  # データベースからカロリー情報を取得する関数をインポート

# FastAPIのインスタンスを作成
app = FastAPI()

# CORS設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*"を使用すると全てのオリジンからのアクセスを許可します。特定のドメインに限定することも可能
    allow_credentials=True,
    allow_methods=["*"],  # 全てのHTTPメソッドを許可
    allow_headers=["*"],  # 全てのHTTPヘッダーを許可
)

# モデルのロード（ResNet50を使用）
from torchvision.models import resnet50, ResNet50_Weights

# ResNet50モデルを定義し、事前学習済みの重みを使用
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# 出力層をFood101のクラス数（101クラス）に合わせて変更
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(num_ftrs, 101)  # Food101のクラス数に合わせる
)

# 訓練済みモデルのパラメータをロード（CPU上でロード）
model.load_state_dict(torch.load(
    'models/best_food101_model.pth',  # 訓練済みモデルのパス
    map_location=torch.device('cpu')  # CPU上でロード
))
model.eval()  # 評価モードに切り替え

# クラスインデックスとクラス名のマッピング（101クラス）
class_names = [
    'アップルパイ', 'ベビーバックリブ', 'バクラバ', '牛肉のカルパッチョ', '牛肉のタルタル',
    'ビートサラダ', 'ベニエ', 'ビビンバ', 'ブレッドプディング', '朝食のブリトー',
    'ブルスケッタ', 'シーザーサラダ', 'カノーリ', 'カプレーゼサラダ', 'キャロットケーキ',
    'セビーチェ', 'チーズケーキ', 'チーズプレート', 'チキンカレー', 'チキンケサディーヤ',
    '手羽先', 'チョコレートケーキ', 'チョコレートムース', 'チュロス', 'クラムチャウダー',
    'クラブサンドイッチ', 'クラブケーキ', 'クレームブリュレ', 'クロックマダム', 'カップケーキ',
    'デビルドエッグ', 'ドーナツ', 'ダンプリング', '枝豆', 'エッグベネディクト',
    'エスカルゴ', 'ファラフェル', 'フィレミニョン', 'フィッシュ＆チップス', 'フォアグラ',
    'フライドポテト', 'フレンチオニオンスープ', 'フレンチトースト', '揚げイカ', 'チャーハン',
    'フローズンヨーグルト', 'ガーリックブレッド', 'ニョッキ', 'ギリシャサラダ', 'グリルチーズサンド',
    '焼き鮭', 'グアカモーレ', '餃子', 'ハンバーガー', '酸辣湯', 'ホットドッグ',
    'ウェボスランチェロス', 'フムス', 'アイスクリーム', 'ラザニア', 'ロブスターのビスク',
    'ロブスターロールサンド', 'マカロニとチーズ', 'マカロン', 'みそ汁', 'ムール貝', 
    'ナチョス', 'オムレツ', 'オニオンリング', '牡蠣', 'パッタイ', 'パエリア', 'パンケーキ',
    'パンナコッタ', '北京ダック', 'フォー', 'ピザ', 'ポークチョップ', 'プーティン', 'プライムリブ',
    'プルドポークサンド', 'ラーメン', 'ラビオリ', 'レッドベルベットケーキ', 'リゾット', 'サモサ',
    '刺身', 'ホタテ', '海藻サラダ', 'エビとグリッツ', 'スパゲティボロネーゼ', 'スパゲッティカルボナーラ',
    '春巻き', 'ステーキ', 'イチゴのショートケーキ', '寿司', 'タコス', 'たこ焼き', 'ティラミス',
    'マグロのタルタル', 'ワッフル'
]

# 画像の前処理（モデル訓練時と同じ設定にする）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 一貫性を持たせるために同じサイズにする
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ルートエンドポイント（APIが稼働しているか確認するためのエンドポイント）
@app.get("/")
async def root():
    return {"message": "API is running. Use /predict to make predictions."}

# 画像から料理名を推論するAPIエンドポイント
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # アップロードされた画像をPIL形式に変換
        img = Image.open(file.file).convert('RGB')
    except Exception as e:
        # 画像が無効な場合にエラーメッセージを返す
        return {"error": f"無効な画像フォーマットです: {str(e)}"}

    # 前処理を適用し、バッチ次元を追加
    img = transform(img).unsqueeze(0)

    # 推論を行い、各クラスの確率を計算
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()

    # 予測されたクラス名を取得
    predicted_class = class_names[outputs.argmax(1).item()]

    # カロリー情報をデータベースから取得
    calorie_info = get_food_calories(predicted_class)

    # 結果をJSON形式で返す
    return {
        "predicted_class": predicted_class,  # 推論された料理名
        "calories": calorie_info,  # カロリー情報を追加
        "probabilities": probabilities,  # 各クラスの確率
        "class_names": class_names  # クラス名のリスト
    }

# アプリを起動するためのエントリーポイント
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # 環境変数からポートを取得
    uvicorn.run(app, host="0.0.0.0", port=port)  # 指定されたポートでアプリを実行
