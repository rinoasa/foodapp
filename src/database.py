import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# SQLiteのデータベースファイルのパス
DATABASE_URL = "sqlite:///./food_app.db"

# データベースファイルが存在しない場合に作成する
if not os.path.exists('food_app.db'):
    open('food_app.db', 'w').close()

# SQLAlchemyのエンジンを作成
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# モデルのベースクラスを定義
Base = declarative_base()

# 食品カロリーテーブルの定義
class FoodCalories(Base):
    __tablename__ = "food_calories"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String, unique=True, index=True)  # 食品名
    calories_per_100g = Column(Float)  # 100gあたりのカロリー

# 食べたもの日記テーブルの定義
class FoodDiary(Base):
    __tablename__ = "food_diary"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String, index=True)  # 食品名
    date = Column(String)  # 日付（YYYY-MM-DD形式）
    mealtime = Column(String)  # 食事の時間帯（例: 朝、昼、夕、間食）
    time = Column(String)  # 時間（HH:MM:SS形式）
    image_path = Column(String)  # 画像の保存パス
    calories = Column(Float)  # 摂取カロリー

# テーブルをデータベース内に作成
Base.metadata.create_all(bind=engine)

# セッションを作成するための設定
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# CSVから食品データを読み込み、DBに登録する関数
def load_food_data_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    db = SessionLocal()
    try:
        for index, row in df.iterrows():
            food_name = row['食品名']  # CSVのカラム名に合わせて修正
            calories = row['カロリー (kcal/100g)']  # CSVのカラム名に合わせて修正

            # 食品が既に存在するか確認
            existing_food = db.query(FoodCalories).filter(FoodCalories.food_name == food_name).first()
            if existing_food:
                print(f"{food_name}は既に存在します。")
                continue

            # 新しい食品のカロリー情報を追加
            food_calorie = FoodCalories(food_name=food_name, calories_per_100g=calories)
            db.add(food_calorie)
            db.commit()
            db.refresh(food_calorie)
            print(f"{food_name}のカロリー情報を追加しました: {calories} kcal")
    except Exception as e:
        print(f"データの追加に失敗しました: {e}")
    finally:
        db.close()

# 指定した食品のカロリー情報を取得する関数
def get_food_calories(food_name: str):
    db = SessionLocal()
    try:
        food_calorie = db.query(FoodCalories).filter(FoodCalories.food_name == food_name).first()
        return food_calorie
    finally:
        db.close()

# 食べたもの日記にエントリを追加する関数
def add_food_diary(food_name: str, date: str, mealtime: str, time: str, image_path: str, calories: float):
    db = SessionLocal()
    try:
        food_diary = FoodDiary(food_name=food_name, date=date, mealtime=mealtime, time=time, image_path=image_path, calories=calories)
        db.add(food_diary)
        db.commit()
        db.refresh(food_diary)
        print(f"食べたもの日記に'{food_name}'を追加しました。")
    except Exception as e:
        print(f"食べたもの日記の追加に失敗しました: {e}")
    finally:
        db.close()

# すべての食べたもの日記を取得する関数
def get_food_diary():
    db = SessionLocal()
    try:
        return db.query(FoodDiary).all()  # すべての食べたもの日記を取得
    finally:
        db.close()

# メイン処理
if __name__ == "__main__":
    csv_path = './data/csv/food_data.csv'  # CSVファイルのパス
    
    # CSVファイルが存在するか確認し、存在する場合はデータをDBに読み込む
    if not os.path.exists(csv_path):
        print(f"CSVファイルが見つかりません: {csv_path}")
    else:
        load_food_data_from_csv(csv_path)  # CSVからデータを読み込み、DBに登録する
