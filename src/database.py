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

# 食品カロリーテーブル
class FoodCalories(Base):
    __tablename__ = "food_calories"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String, unique=True, index=True)
    calories_per_100g = Column(Float)

# 食べたもの日記テーブル
class FoodDiary(Base):
    __tablename__ = "food_diary"

    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String, index=True)
    date = Column(String)  # 'YYYY-MM-DD'形式
    mealtime = Column(String)
    time = Column(String)  # 'HH:MM:SS'形式
    image_path = Column(String)
<<<<<<< HEAD
    calories = Column(Float)  # カロリー情報を追加
=======
>>>>>>> 91ac4088b (Initial commit with Git LFS for large files)

# テーブルを作成
Base.metadata.create_all(bind=engine)

# セッションの作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# CSVからデータを読み込み、DBに登録する関数
def load_food_data_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
<<<<<<< HEAD
=======
    # セッションを作成
>>>>>>> 91ac4088b (Initial commit with Git LFS for large files)
    db = SessionLocal()
    try:
        for index, row in df.iterrows():
            food_name = row['食品名']  # CSVのカラム名に合わせて修正
            calories = row['カロリー (kcal/100g)']  # CSVのカラム名に合わせて修正

<<<<<<< HEAD
=======
            # 既に存在するか確認
>>>>>>> 91ac4088b (Initial commit with Git LFS for large files)
            existing_food = db.query(FoodCalories).filter(FoodCalories.food_name == food_name).first()
            if existing_food:
                print(f"{food_name}は既に存在します。")
                continue

<<<<<<< HEAD
=======
            # データを追加
>>>>>>> 91ac4088b (Initial commit with Git LFS for large files)
            food_calorie = FoodCalories(food_name=food_name, calories_per_100g=calories)
            db.add(food_calorie)
            db.commit()
            db.refresh(food_calorie)
            print(f"{food_name}のカロリー情報を追加しました: {calories} kcal")
    except Exception as e:
        print(f"データの追加に失敗しました: {e}")
    finally:
        db.close()

# カロリー情報を取得する関数
def get_food_calories(food_name: str):
    db = SessionLocal()
    try:
        food_calorie = db.query(FoodCalories).filter(FoodCalories.food_name == food_name).first()
        return food_calorie
    finally:
        db.close()

<<<<<<< HEAD
# 食べたもの日記を追加する関数
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

# 食べたもの日記を取得する関数
def get_food_diary():
    db = SessionLocal()
    try:
        return db.query(FoodDiary).all()  # すべての食べたもの日記を取得
    finally:
        db.close()

=======
>>>>>>> 91ac4088b (Initial commit with Git LFS for large files)
# メイン処理
if __name__ == "__main__":
    csv_path = './data/csv/food_data.csv'  # CSVファイルのパス
    
<<<<<<< HEAD
=======
    # CSVファイルの存在を確認
>>>>>>> 91ac4088b (Initial commit with Git LFS for large files)
    if not os.path.exists(csv_path):
        print(f"CSVファイルが見つかりません: {csv_path}")
    else:
        load_food_data_from_csv(csv_path)  # CSVからデータを読み込み、DBに登録する
