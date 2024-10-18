from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import models, transforms
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from src.database import get_food_calories  # データベースからカロリー情報を取得する関数をインポート

app = FastAPI()

# CORS設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*"を使用すると全てのオリジンからのアクセスを許可します。特定のドメインに限定することも可能。
    allow_credentials=True,
    allow_methods=["*"],  # 全てのHTTPメソッドを許可
    allow_headers=["*"],  # 全てのHTTPヘッダーを許可
)

# モデルのロード（ResNet50を使用）
from torchvision.models import resnet50, ResNet50_Weights

# ResNet50モデルを定義し、事前学習済みの重みを使用
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# 出力層を訓練時と同じ構造に変更（101クラスに合わせる）
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(num_ftrs, 101)  # Food101のクラス数に合わせる
)

# 訓練済みモデルのパラメータをロード（CPU上でロード）
model.load_state_dict(torch.load(
    'models/best_food101_model.pth',  # 訓練済みモデルのパス
    map_location=torch.device('cpu'),
    weights_only=True  # 安全性を高めるためにオプションを追加
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

# 画像の前処理（訓練時と同じ設定）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 画像から料理名を推論するAPIエンドポイント
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # アップロードされた画像をPIL形式に変換
        img = Image.open(file.file).convert('RGB')
    except Exception as e:
        return {"error": f"無効な画像フォーマットです: {str(e)}"}

    # 前処理を適用してバッチ次元を追加
    img = transform(img).unsqueeze(0)

    # 推論（勾配計算は不要）
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()  # 各クラスの確率を取得

    # 予測されたクラス名とその確率を取得
    predicted_class = class_names[outputs.argmax(1).item()]

    # カロリー情報を取得
    calorie_info = get_food_calories(predicted_class)

    return {
        "predicted_class": predicted_class,
        "calories": calorie_info,  # カロリー情報を追加
        "probabilities": probabilities,  # 各クラスの確率をリストとして返す
        "class_names": class_names        # クラス名のリストを返す
    }
