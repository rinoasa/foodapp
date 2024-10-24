import streamlit as st
import requests
from PIL import Image
import io
import torch
from datetime import datetime, time
import time as time_module
from database import add_food_diary, get_food_calories, get_food_diary

# サイドバーでAPIエンドポイントを選択する
st.sidebar.title("環境設定")
environment = st.sidebar.selectbox("API環境を選択", ["ローカル", "Render"])

# FastAPIのエンドポイントを環境に応じて設定
API_URL = "http://127.0.0.1:8000/predict" if environment == "ローカル" else "https://foodapp-frj7.onrender.com/predict"

# "食べたもの"を保存するリストをセッションに保存
if 'eaten_foods' not in st.session_state:
    st.session_state.eaten_foods = []

# ページングのための初期化
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0

# "時間帯"の初期化
if 'selected_mealtime' not in st.session_state:
    st.session_state.selected_mealtime = "朝"

# "時間"の初期化
if 'selected_time' not in st.session_state:
    st.session_state.selected_time = time(8, 0)  # デフォルトの時間

# "classification_result"の初期化
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

# "uploaded_file"の初期化
uploaded_file = None

# 1ページに表示するアイテム数
ITEMS_PER_PAGE = 9

# ページ数を計算
def get_total_pages(items, items_per_page):
    return max(1, ((len(items) - 1) // items_per_page + 1))

# 指定されたページ番号に応じて表示するアイテムを取得
def get_page_items(items, page_number, items_per_page):
    start_idx = page_number * items_per_page
    end_idx = start_idx + items_per_page
    return items[start_idx:end_idx]

# ページの切り替えボタンを表示
def render_pagination_controls(page_number, total_pages):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("前へ", key=f"prev_{page_number}", disabled=page_number <= 0):
            st.session_state.page_number -= 1
    with col2:
        st.write(f"ページ {page_number + 1} / {total_pages}")
    with col3:
        if st.button("次へ", key=f"next_{page_number}", disabled=page_number >= total_pages - 1):
            st.session_state.page_number += 1

# サイドバーで画像アップロード機能を表示
with st.sidebar:
    st.header("画像アップロード")
    uploaded_file = st.file_uploader("画像を選択...", type=["jpg", "jpeg", "png"])

# タブの作成
tab1, tab2 = st.tabs(["食事分類", "食べたもの日記"])

# タブ1: 食事分類ページ
with tab1:
    st.title("Food Classification App")
    st.write("サイドバーから画像をアップロードし、「食事を分類する」ボタンを押してください。")

    if uploaded_file is not None:
        # 画像を表示 (さらに小さく表示)
        image = Image.open(uploaded_file)
        img_width, img_height = image.size
        canvas_size = max(img_width, img_height)
        new_image = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
        paste_position = (
            (canvas_size - img_width) // 2,
            (canvas_size - img_height) // 2
        )
        new_image.paste(image, paste_position)
        st.image(new_image, caption='アップロードした画像', use_column_width=False, width=200)

        # 分類ボタンを押すと分類処理が開始
        if st.button("食事を分類する"):
            # プログレスバーの初期化
            progress_bar = st.progress(0)

            # プログレスバーを進める（分類中の進行をシミュレート）
            for i in range(100):
                time_module.sleep(0.01)
                progress_bar.progress(i + 1)

            # 画像をFastAPIに送信する
            with st.spinner("推定中..."):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                # FastAPIにリクエストを送信
                response = requests.post(API_URL, files={"file": ('image.jpg', img_byte_arr, 'image/jpeg')})

                # エラーチェック
                if response.status_code == 200:
                    result = response.json()
                    if 'probabilities' in result and 'class_names' in result:
                        probabilities = result['probabilities']
                        class_names = result['class_names']

                        top5_probs, top5_idx = torch.topk(torch.tensor(probabilities), 5)
                        top5_classes = [class_names[i] for i in top5_idx]
                        top5_confidences = [f"{probabilities[i] * 100:.2f}%" for i in top5_idx]

                        st.session_state.classification_result = {
                            "top5_classes": top5_classes,
                            "top5_confidences": top5_confidences,
                            "uploaded_file": uploaded_file
                        }

                        progress_bar.empty()
                        st.success("分類が完了しました。結果を以下に表示します。")

                        st.write("**上位5位の推定結果:**")
                        for food, confidence in zip(top5_classes, top5_confidences):
                            st.write(f"- {food} : {confidence}")

                else:
                    st.error("画像の分類に失敗しました。サーバーに問題がある可能性があります。")
                    progress_bar.empty()

    # ランキング結果が保存されている場合に表示し続ける
    if st.session_state.get("classification_result"):
        # 食べたものを選択するリストボックス
        other_food = st.checkbox("それ以外の料理を登録する")

        if other_food:
            selected_food = st.text_input("料理名を入力してください")
        else:
            selected_food = st.selectbox("食べたものを選んでください", st.session_state.classification_result["top5_classes"])

        # グラム数の入力
        grams = st.number_input("グラム数を入力してください", min_value=0, value=100)

        # カロリー計算ボタン
        if st.button("カロリーを計算する"):
            calorie_info = get_food_calories(selected_food)
            if calorie_info:
                calories_per_100g = calorie_info.calories_per_100g
                total_calories = (calories_per_100g * grams) / 100
                st.session_state.total_calories = total_calories
                st.write(f"**カロリー情報:** {selected_food}: {total_calories:.2f} kcal")
            else:
                st.error("カロリー情報が見つかりませんでした。")

        selected_date = st.date_input("日付", datetime.now())
        mealtime = st.selectbox("時間帯", ["朝", "昼", "夕", "間食"], key="mealtime_select")

        # 時間帯選択によってデフォルト時間を設定
        if mealtime == "朝":
            st.session_state.selected_time = time(8, 0)
        elif mealtime == "昼":
            st.session_state.selected_time = time(12, 0)
        elif mealtime == "夕":
            st.session_state.selected_time = time(18, 0)
        else:
            st.session_state.selected_time = time(15, 0)  # 間食のデフォルト時間

        selected_time = st.time_input("時間", st.session_state.selected_time)

        # 「食べたものとして追加」ボタン
        if st.button("食べたものとして追加"):
            if 'total_calories' in st.session_state:
                image_path = f"uploaded_images/{uploaded_file.name}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                add_food_diary(selected_food, str(selected_date), st.session_state.selected_mealtime, str(selected_time), image_path, st.session_state.total_calories)
                st.success(f"'{selected_food}'を食べたものに追加しました。")
            else:
                st.error("カロリー情報が計算されていません。")

# タブ2: 食べたもの日記ページ
with tab2:
    st.title("食べたもの日記")
    food_diaries = get_food_diary()

    if food_diaries:
        total_pages = get_total_pages(food_diaries, ITEMS_PER_PAGE)
        page_items = get_page_items(food_diaries, st.session_state.page_number, ITEMS_PER_PAGE)

        cols = st.columns(3)  # 3カラムのグリッドを作成
        for idx, diary in enumerate(page_items):
            with cols[idx % 3]:  # カラムを選択
                # 画像を1:1の正方形で表示
                image = Image.open(diary.image_path)
                img_width, img_height = image.size
                max_size = max(img_width, img_height)
                new_image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
                new_image.paste(image, ((max_size - img_width) // 2, (max_size - img_height) // 2))
                st.image(new_image, caption=f"{diary.food_name}", use_column_width=True)

                st.write(f"**料理名**: {diary.food_name}")
                st.write(f"**日付**: {diary.date}")
                st.write(f"**時間帯**: {diary.mealtime}")
                st.write(f"**時間**: {diary.time[:5]}")  # hh:mm形式に修正
                st.write(f"**カロリー**: {diary.calories} kcal")

        render_pagination_controls(st.session_state.page_number, total_pages)
    else:
        st.write("日記はまだありません。")
