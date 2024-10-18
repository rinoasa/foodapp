
# 食事分類モデル (FoodApp)

このプロジェクトは、食事分類モデルを使用して画像を分類するアプリケーションです。ユーザーは食事の画像をアップロードし、その画像が何の料理であるかを特定し、カロリー情報も取得することができます。

## フォルダ構成

```
FOODAPP                        
│
├── .streamlit                 # Streamlitの設定ファイル
│   └── config.toml           
│
├── data                       # データ関連
│   └── csv                    # CSVファイルを格納
│       └── food_data.csv      # 食品データのCSV
│
├── models                     # モデル関連
│   └── best_food101_model.pth  # 学習済みモデル
│
├── src                        # ソースコード
│   ├── __pycache__           
│   ├── __init__.py           
│   ├── app.py                # アプリケーションのメインロジック
│   ├── database.py           # データベース操作
│   ├── main.py               # アプリケーションのエントリポイント
│   └── model.py              # モデル定義
│
├── test                       # テスト関連
│   ├── accuracy_plot.png      # 精度のプロット画像
│   ├── loss_plot.png          # ロスのプロット画像
│   └── training_log.txt       # 学習ログ
│
├── hoge                       # 個人データ用のディレクトリ
│   ├── hoge1.jpg              
│   ├── hoge2.jpg              
│   └── hoge3.jpg              
│
├── venv                       # 仮想環境
│
├── .gitignore                 # Gitで無視するファイルの設定
│
├── food_app.db                # SQLiteデータベース
│
├── README.md                  # プロジェクトの説明
│
└── requirements.txt           # 依存パッケージリスト
```

## インストール

1. **リポジトリをクローン**します。
   ```bash
   git clone <リポジトリのURL>
   ```

2. **仮想環境を作成**します。  
   Pythonの仮想環境を作成して有効化します。

   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **必要なパッケージをインストール**します。
   ```bash
   pip install -r requirements.txt
   ```
## アプリ画面のイメージ

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/rinoasa/foodapp/blob/dev/test/foodapp_sample_image1.png?raw=true" alt="Streamlit App Interface 1" style="width: 45%;"/>
    <img src="https://github.com/rinoasa/foodapp/blob/dev/test/foodapp_sample_image2.png?raw=true" alt="Streamlit App Interface 2" style="width: 45%;"/>
</div>

## データベースの管理

- 食品名とカロリーのデータベースは`food_data.csv`を基に初期化されます。`src/database.py`を実行して、CSVデータをデータベースに登録します。

## 使用方法

1. **FastAPIサーバーを起動**します。
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Streamlitアプリを起動**します。
   ```bash
   streamlit run src/app.py
   ```

3. **ブラウザでアプリにアクセス**します。  
   [http://localhost:8501](http://localhost:8501) にアクセスし、食事画像をアップロードして分類を行います。
   

## ライセンス

このプロジェクトはライセンスが設定されていません。
