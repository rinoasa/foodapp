# 必要なライブラリをインポート
import streamlit as st  # Streamlitライブラリのインポート（UIを作成）
import requests  # APIとの通信のため
from PIL import Image  # 画像操作のため
import io  # バイナリデータの操作
import torch  # PyTorchライブラリ（深層学習のため）
from datetime import datetime, time, timedelta  # 日付と時間の操作
import time as time_module  # 時間制御
import pandas as pd  # データフレーム操作
import plotly.graph_objects as go  # グラフ作成のためのPlotlyライブラリ
from database import add_food_diary, get_food_calories, get_food_diary  # データベース操作のための関数をインポート

# FastAPIのエンドポイントURLを指定
API_URL = "http://127.0.0.1:8000/predict"

# セッション状態に「食べたもの」リストが保存されていない場合、初期化
if 'eaten_foods' not in st.session_state:
    st.session_state.eaten_foods = []

# 時間帯の初期化（デフォルトは朝）
if 'selected_mealtime' not in st.session_state:
    st.session_state.selected_mealtime = "朝"

# 時間の初期化（デフォルトは8:00）
if 'selected_time' not in st.session_state:
    st.session_state.selected_time = time(8, 0)

# 食事分類結果の初期化
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

# サイドバーで画像アップロード機能とページ切り替え機能を提供
with st.sidebar:
    st.header("画像アップロード")  # サイドバーの見出し
    uploaded_file = st.file_uploader("画像を選択...", type=["jpg", "jpeg", "png"])  # 画像アップロード

    # ページ切り替えのための選択ボックス
    page_selection = st.selectbox("ページを選択", ["食事分類", "食べたもの日記", "カロリーダッシュボード"])

# 食事分類ページの処理
if page_selection == "食事分類":
    st.title("食事分類")  # ページタイトル
    st.write("１．サイドバーから画像をアップロードし、「食事を分類する」ボタンを押してください。")  # ユーザーガイド

    if uploaded_file is not None:
        # アップロードされた画像を表示（画像を中央に配置）
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

        # 食事分類ボタンが押された場合
        if st.button("食事を分類する"):
            # プログレスバーを初期化して進行中のシミュレーションを行う
            progress_bar = st.progress(0)
            for i in range(100):
                time_module.sleep(0.01)
                progress_bar.progress(i + 1)

            # 画像をFastAPIのエンドポイントに送信して推論を実施
            with st.spinner("推定中..."):
                img_byte_arr = io.BytesIO()  # 画像をバイナリ形式に変換
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()

                # FastAPIに画像をPOSTリクエストで送信
                response = requests.post(API_URL, files={"file": ('image.jpg', img_byte_arr, 'image/jpeg')})

                # サーバーからのレスポンスチェック
                if response.status_code == 200:
                    result = response.json()
                    if 'probabilities' in result and 'class_names' in result:
                        probabilities = result['probabilities']
                        class_names = result['class_names']

                        # 推定確率の上位5つを取得し、表示
                        top5_probs, top5_idx = torch.topk(torch.tensor(probabilities), 5)
                        top5_classes = [class_names[i] for i in top5_idx]
                        top5_confidences = [f"{probabilities[i] * 100:.2f}%" for i in top5_idx]

                        # 結果をセッションに保存
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

    # 分類結果が保存されている場合に次のステップを実行
    if st.session_state.get("classification_result"):
        st.write("-------------------------------")
        st.write("２．食べたものとグラム数を入力し、「カロリーを計算する」ボタンを押してください。")
        
        # 食べたものを選択するリストボックス
        other_food = st.checkbox("それ以外の料理を登録する")
        if other_food:
            selected_food = st.text_input("料理名を入力してください")
        else:
            selected_food = st.selectbox("食べたものを選んでください", st.session_state.classification_result["top5_classes"])

        # グラム数の入力（デフォルトは100g）
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
        
        st.write("-------------------------------")
        st.write("３．食事をした日付・時間帯・時間を入力し、「食べたものとして追加」ボタンを押してください。")

        selected_date = st.date_input("日付", datetime.now())
        mealtime = st.selectbox("時間帯", ["朝", "昼", "夕", "間食"], key="mealtime_select")

        # 時間帯選択によってデフォルト時間を設定
        st.session_state.selected_mealtime = mealtime  # 時間帯の更新

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
                # データベースのデータを再読み込み
                st.session_state.food_diaries = get_food_diary()  # DBから日記を再取得

        st.write("-------------------------------")
        st.write("４．サイドバーのページの選択から食べたもの日記・カロリーダッシュボードを開くことができます。")

# 食べたもの日記ページ
elif page_selection == "食べたもの日記":
    st.title("食べたもの日記")
    food_diaries = get_food_diary()

    if food_diaries:
        cols = st.columns(3)  # 3カラムのグリッドを作成
        for idx, diary in enumerate(food_diaries):
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
    else:
        st.write("日記はまだありません。")

# ダッシュボードページ
elif page_selection == "カロリーダッシュボード":
    st.title("カロリーダッシュボード") 

    # 今日と昨日の総カロリーを計算
    today_date = datetime.now().date()
    yesterday_date = today_date - timedelta(days=1)

    # ダイアリーを取得
    food_diaries = get_food_diary()
    today_total = sum(entry.calories for entry in food_diaries if entry.date == today_date.strftime("%Y-%m-%d"))
    yesterday_total = sum(entry.calories for entry in food_diaries if entry.date == yesterday_date.strftime("%Y-%m-%d"))
    calorie_difference = today_total - yesterday_total

    # カロリー差の表示
    st.write("\n")
    st.subheader("Q. 今日は昨日よりどのくらい食べた？", divider="gray")
    st.write(f"本日{today_date} と昨日 {yesterday_date} の総カロリーの差:")
    if calorie_difference > 0:
        st.metric(label="カロリー差", value=f"{calorie_difference:.1f} kcal", delta=f"+{calorie_difference:.1f} kcal")
    else:
        st.metric(label="カロリー差", value=f"{calorie_difference:.1f} kcal", delta=f"{calorie_difference:.1f} kcal")

    # 日付範囲の選択
    st.write("\n")
    st.write("\n")
    st.subheader("Q. 指定の期間において、毎日どのくらい食べた？", divider="gray")
    start_date = st.date_input("開始日", datetime.now() - timedelta(days=30))
    end_date = st.date_input("終了日", datetime.now())

    # 月間ダッシュボード
    month_calorie_summary = {date.day: {"朝": 0, "昼": 0, "夕": 0, "間食": 0} for date in pd.date_range(start=start_date, end=end_date)}
    
    for entry in food_diaries:
        entry_date = datetime.strptime(entry.date, "%Y-%m-%d").date()  # datetime.date型に変換
        if start_date <= entry_date <= end_date:
            month_calorie_summary[entry_date.day][entry.mealtime] += entry.calories

    # 積み上げ棒グラフの作成
    month_labels = list(month_calorie_summary.keys())
    breakfast_values = [month_calorie_summary[day]["朝"] for day in month_labels]
    lunch_values = [month_calorie_summary[day]["昼"] for day in month_labels]
    dinner_values = [month_calorie_summary[day]["夕"] for day in month_labels]
    snack_values = [month_calorie_summary[day]["間食"] for day in month_labels]

    # Plotlyを使用したグラフの作成
    fig = go.Figure()

    # 各食事の積み上げ
    fig.add_trace(go.Bar(
        x=month_labels,
        y=breakfast_values,
        name='朝',
        marker_color='#FF9999',
        hovertemplate='朝: %{y} kcal<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=month_labels,
        y=lunch_values,
        name='昼',
        marker_color='#66B3FF',
        hovertemplate='昼: %{y} kcal<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=month_labels,
        y=dinner_values,
        name='夕',
        marker_color='#99FF99',
        hovertemplate='夕: %{y} kcal<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        x=month_labels,
        y=snack_values,
        name='間食',
        marker_color='#FFD700',
        hovertemplate='間食: %{y} kcal<extra></extra>'
    ))

    # 総カロリーの計算
    total_calories = [sum([breakfast_values[i], lunch_values[i], dinner_values[i], snack_values[i]]) for i in range(len(month_labels))]

    # 総カロリーの線を追加
    fig.update_layout(
        barmode='stack',
        title=f"{start_date} から {end_date} までの日ごとのカロリー合計",
        xaxis_title="日",
        yaxis_title="カロリー (kcal)",
        yaxis=dict(tickprefix=''),
    )

    st.plotly_chart(fig)

    # 日ごとのダッシュボード
    st.write("\n")
    st.write("\n")
    st.subheader("Q. 指定の日付において、どのくらい食べた？", divider="gray")
    selected_date = st.date_input("日付を選択", datetime.now())  # デフォルトは今日の日付

    # 日ごとのカロリー集計
    daily_calorie_summary = {
        "朝": 0,
        "昼": 0,
        "夕": 0,
        "間食": 0,
    }

    for entry in food_diaries:
        entry_date = datetime.strptime(entry.date, "%Y-%m-%d").date()  # datetime.date型に変換
        if entry_date == selected_date:
            daily_calorie_summary[entry.mealtime] += entry.calories

    # 総カロリーの表示
    total_calories = sum(daily_calorie_summary.values())
    st.subheader(f"{selected_date} の総カロリー: {total_calories} kcal")

    # 棒グラフの作成
    daily_labels = list(daily_calorie_summary.keys())
    daily_values = list(daily_calorie_summary.values())

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily_labels, y=daily_values, marker_color=['#FF9999', '#66B3FF', '#99FF99', '#FFD700']))

    fig2.update_layout(title=f"{selected_date} のカロリー内訳", xaxis_title="食事の種類", yaxis_title="カロリー (kcal)", yaxis=dict(tickprefix=''))
    
    st.plotly_chart(fig2)
