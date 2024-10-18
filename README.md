
# 食事分類モデル (FoodApp)

このプロジェクトは、食事分類モデルを使用して画像を分類するアプリケーションです。ユーザーは食事の画像をアップロードし、その画像が何の料理であるかを特定し、カロリー情報も取得することができます。

## フォルダ構成

```
FOOD_APP/
├── .streamlit/                    # Streamlitの設定ファイル
│   └── config.toml                # Streamlitのテーマ設定
├── data/                          # データ関連
│   ├── csv/                       # カロリー情報が含まれたCSVファイル
│   │   └── food_data.csv          # 食品名とカロリー情報
│   ├── food-101/                  # Food101データセット
│   │   ├── images/                # 画像データ
│   │   ├── meta/                  # メタデータ
│   │   ├── license_agreement.txt  # ライセンス情報
│   │   └── README.txt             # データセットのREADME
│   └── processed/                 # 処理済みデータ（圧縮ファイルなど）
│       └── food-101.tar.gz        # 処理されたデータ
├── models/                        # モデル関連
│   └── best_food101_model.pth     # 訓練済みモデル
├── src/                           # ソースコード
│   ├── app.py                     # Streamlitアプリのエントリポイント
│   ├── database.py                # データベース操作を行うモジュール
│   ├── main.py                    # FastAPIアプリのエントリポイント
│   └── model.py                   # モデル定義とトレーニングロジック
├── test/                          # テスト関連ファイル
│   ├── accuracy_plot.png          # 精度プロット画像
│   ├── loss_plot.png              # 損失プロット画像
│   └── training_log.txt           # トレーニングログ
├── uploaded_images/               # ユーザーがアップロードした画像（無視される）
├── .gitignore                     # Gitが無視するファイルリスト
├── food_app.db                    # SQLiteデータベースファイル（無視される）
├── README.md                      # このファイル
└── requirements.txt               # 依存関係リスト
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

4. **Food-101データセットをダウンロード**し、展開されていない場合は`data/processed/`ディレクトリ内に`food-101.tar.gz`を保存し、必要に応じて展開します。

5. **`food_data.csv`をデータベースにインポート**します（必要に応じて）。
   ```bash
   python src/database.py
   ```

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

## データベースの管理

- 食品名とカロリーのデータベースは`food_data.csv`を基に初期化されます。`src/database.py`を実行して、CSVデータをデータベースに登録します。

## ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。
