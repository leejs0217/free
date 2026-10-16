import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

st.title("🎬 영화 유형 나누기")
st.caption(
    "영화의 스크린 수, 누적 관객, 10위권 일수, "
    "롱런 지수를 바탕으로 비슷한 영화끼리 묶습니다."
)


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {', '.join(missing_columns)}"
        )

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["movieNm"] = df["movieNm"].fillna("제목 없음").astype(str)

    # 원본 데이터의 전체 편수
    total_count = len(df)

    # 첫 주 관객이 0 이하인 영화와 필수 수치 결측 영화 제외
    df = df.dropna(subset=numeric_columns).copy()
    df = df[df["first_week_audi"] > 0].copy()

    # 상용로그는 양수에서만 정의되므로 스크린 수와 누적 관객도
    # 양수인 영화만 분석에 사용
    df = df[
        (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
    ].copy()

    # 원래 단위의 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    # 군집화용 로그 변환 속성
    df["log_first_scrn"] = df["first_scrn"].map(math.log10)
    df["log_total_audi"] = df["total_audi"].map(math.log10)

    # 네 가지 군집화 속성
    feature_labels = {
        "log_first_scrn": "스크린 수(상용로그)",
        "log_total_audi": "누적 관객(상용로그)",
        "days_in_top10": "10위권 일수",
        "long_run_index": "롱런 지수",
    }

    # 원래 단위의 요약표에 사용할 속성
    original_labels = {
        "first_scrn": "스크린 수",
        "total_audi": "누적 관객",
        "days_in_top10": "10위권 일수",
        "long_run_index": "롱런 지수",
    }

    # 계산에 사용되는 속성에 결측치가 없는 영화만 남김
    df = df.dropna(subset=list(feature_labels)).copy()

    return df, total_count, feature_labels, original_labels


try:
    df, total_count, feature_labels, original_labels = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.stop()

if df.empty:
    st.warning("분석할 수 있는 영화가 없습니다.")
    st.stop()


# --------------------------------------------------
# 군집화 속성 선택
# --------------------------------------------------
st.subheader("1. 군집화 속성 선택")

all_features = list(feature_labels.keys())

selected_features = st.multiselect(
    "군집을 나누는 데 사용할 속성을 선택하세요.",
    options=all_features,
    default=all_features,
    format_func=lambda x: feature_labels[x],
)

if not selected_features:
    st.warning("군집화 속성을 하나 이상 선택하세요.")
    st.stop()


# --------------------------------------------------
# 표준화 및 K-평균 군집화
# --------------------------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[selected_features])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
df = df.copy()
df["cluster_id"] = kmeans.fit_predict(scaled_features)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    df.groupby("cluster_id")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_names = {
    cluster_id: chr(ord("㉮") + rank)
    for rank, cluster_id in enumerate(cluster_order)
}
df["군집"] = df["cluster_id"].map(cluster_names)

cluster_color_order = ["㉮", "㉯", "㉰"]
color_map = {
    "㉮": "#E45756",
    "㉯": "#4C78A8",
    "㉰": "#54A24B",
}

# 전체 편수와 실제 군집화에 사용된 편수
st.markdown(
    f"**전체 영화: {total_count:,}편　|　"
    f"군집화한 영화: {len(df):,}편**"
)

st.caption(
    "결측치가 있거나 첫 주 관객이 0 이하인 영화, "
    "로그 변환이 불가능한 스크린 수 또는 누적 관객이 "
    "있는 영화는 분석에서 제외합니다."
)


# --------------------------------------------------
# 2차원 산점도
# --------------------------------------------------
st.subheader("2. 2차원 산점도")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=all_features,
        index=0,
        format_func=lambda x: feature_labels[x],
        key="scatter_x",
    )

with axis_col2:
    y_feature = st.selectbox(
        "세로축 속성",
        options=all_features,
        index=1,
        format_func=lambda x: feature_labels[x],
        key="scatter_y",
    )

scatter_df = df.copy()
scatter_df["영화 제목"] = scatter_df["movieNm"]
scatter_df["군집"] = pd.Categorical(
    scatter_df["군집"],
    categories=cluster_color_order,
    ordered=True,
)

fig_2d = px.scatter(
    scatter_df,
    x=x_feature,
    y=y_feature,
    color="군집",
    color_discrete_map=color_map,
    category_orders={"군집": cluster_color_order},
    hover_name="영화 제목",
    hover_data={
        "군집": True,
        x_feature: ":.3f",
        y_feature: ":.3f",
    },
    labels={
        x_feature: feature_labels[x_feature],
        y_feature: feature_labels[y_feature],
        "군집": "영화 유형",
    },
    title="속성별 영화 군집 분포",
)

fig_2d.update_traces(marker={"size": 7, "opacity": 0.75})
fig_2d.update_layout(
    height=560,
    legend_title_text="영화 유형",
    margin={"l": 20, "r": 20, "t": 65, "b": 20},
)

st.plotly_chart(fig_2d, use_container_width=True)


# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------
st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 표시하려면 군집화 속성을 "
        "3개 이상 선택하세요."
    )
else:
    axis_x, axis_y, axis_z = st.columns(3)

    with axis_x:
        x3 = st.selectbox(
            "X축 속성",
            options=selected_features,
            format_func=lambda x: feature_labels[x],
            key="scatter3_x",
        )

    with axis_y:
        y3 = st.selectbox(
            "Y축 속성",
            options=selected_features,
            index=min(1, len(selected_features) - 1),
            format_func=lambda x: feature_labels[x],
            key="scatter3_y",
        )

    with axis_z:
        z3 = st.selectbox(
            "Z축 속성",
            options=selected_features,
            index=min(2, len(selected_features) - 1),
            format_func=lambda x: feature_labels[x],
            key="scatter3_z",
        )

    fig_3d = px.scatter_3d(
        scatter_df,
        x=x3,
        y=y3,
        z=z3,
        color="군집",
        color_discrete_map=color_map,
        category_orders={"군집": cluster_color_order},
        hover_name="영화 제목",
        hover_data={
            "군집": True,
            x3: ":.3f",
            y3: ":.3f",
            z3: ":.3f",
        },
        labels={
            x3: feature_labels[x3],
            y3: feature_labels[y3],
            z3: feature_labels[z3],
            "군집": "영화 유형",
        },
        title="3차원 영화 군집 분포",
    )

    fig_3d.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )
    fig_3d.update_layout(
        height=650,
        legend_title_text="영화 유형",
        margin={"l": 0, "r": 0, "t": 65, "b": 0},
        scene={
            "xaxis_title": feature_labels[x3],
            "yaxis_title": feature_labels[y3],
            "zaxis_title": feature_labels[z3],
        },
    )

    st.plotly_chart(fig_3d, use_container_width=True)


# --------------------------------------------------
# 군집별 편수 및 원래 단위의 평균
# --------------------------------------------------
st.subheader("4. 군집별 요약")

summary_rows = []

for cluster_name in cluster_color_order:
    group = df[df["군집"] == cluster_name]

    if group.empty:
        continue

    summary_rows.append(
        {
            "군집": cluster_name,
            "영화 편수": len(group),
            "스크린 수 평균": group["first_scrn"].mean(),
            "누적 관객 평균": group["total_audi"].mean(),
            "10위권 일수 평균": group["days_in_top10"].mean(),
            "롱런 지수 평균": group["long_run_index"].mean(),
        }
    )

summary_df = pd.DataFrame(summary_rows)

st.dataframe(
    summary_df.style.format(
        {
            "영화 편수": "{:,.0f}",
            "스크린 수 평균": "{:,.1f}",
            "누적 관객 평균": "{:,.0f}",
            "10위권 일수 평균": "{:,.1f}",
            "롱런 지수 평균": "{:,.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 군집별 누적 관객 상위 영화 5편
# --------------------------------------------------
st.subheader("5. 군집별 누적 관객 상위 영화")

for cluster_name in cluster_color_order:
    group = (
        df[df["군집"] == cluster_name]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    if group.empty:
        continue

    with st.expander(
        f"군집 {cluster_name} — 누적 관객 상위 영화",
        expanded=True,
    ):
        top_movies = group[["movieNm", "total_audi"]].copy()
        top_movies.columns = ["영화 제목", "누적 관객"]

        top_movies.insert(
            0,
            "순위",
            range(1, len(top_movies) + 1),
        )

        st.dataframe(
            top_movies.style.format(
                {"누적 관객": "{:,.0f}명"}
            ),
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    "군집은 선택한 속성과 표준화 방식에 따라 달라질 수 있습니다. "
    "난수 시드를 고정해 동일한 데이터와 설정에서 결과를 "
    "재현할 수 있도록 했습니다."
)
