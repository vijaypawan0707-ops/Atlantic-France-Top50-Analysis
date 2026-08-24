import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="France Top 50 Playlist Analysis", page_icon="🎵", layout="wide")
st.title("🎵 France Top 50 Playlist Analysis")
st.caption("Audience Sensitivity, Content Compliance & Format Preference Analysis")

@st.cache_data
def load_data():
    file_path = Path("Atlantic_France.csv")
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)

df = load_data()

if df is None:
    st.error("Atlantic_France.csv not found. Please upload it to the same GitHub repository as app.py.")
    st.stop()

df = df.drop_duplicates()
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

for col in ["position", "popularity", "duration_ms", "total_tracks"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["duration_min"] = df["duration_ms"] / 60000
df["album_type"] = df["album_type"].astype(str).str.strip().str.lower()

if df["is_explicit"].dtype != bool:
    df["is_explicit"] = (
        df["is_explicit"].astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

df["rank_tier"] = pd.cut(
    df["position"],
    bins=[0, 10, 25, 50],
    labels=["Top 10", "Top 25", "Top 50"],
    include_lowest=True
)

df["duration_bucket"] = pd.cut(
    df["duration_min"],
    bins=[0, 3, 4, np.inf],
    labels=["Short (<3 min)", "Medium (3–4 min)", "Long (>4 min)"],
    include_lowest=True
)

df["content_acceptance_score"] = ((51 - df["position"]) / 50) * 100

st.success(f"Dataset loaded successfully | Rows: {len(df):,}")

st.sidebar.header("Filters")
filtered_df = df.copy()

if df["date"].notna().any():
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_date) &
            (filtered_df["date"].dt.date <= end_date)
        ]

rank_options = [str(x) for x in df["rank_tier"].dropna().unique()]
selected_ranks = st.sidebar.multiselect("Rank tier", rank_options, default=rank_options)
if selected_ranks:
    filtered_df = filtered_df[filtered_df["rank_tier"].astype(str).isin(selected_ranks)]

album_options = sorted(df["album_type"].dropna().unique().tolist())
selected_album_types = st.sidebar.multiselect("Album type", album_options, default=album_options)
if selected_album_types:
    filtered_df = filtered_df[filtered_df["album_type"].isin(selected_album_types)]

explicit_filter = st.sidebar.selectbox("Explicit status", ["All", "Explicit", "Clean"])
if explicit_filter == "Explicit":
    filtered_df = filtered_df[filtered_df["is_explicit"] == True]
elif explicit_filter == "Clean":
    filtered_df = filtered_df[filtered_df["is_explicit"] == False]

st.subheader("Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Explicit Content Share", f"{filtered_df['is_explicit'].mean()*100:.2f}%")
c2.metric("Single Share", f"{(filtered_df['album_type']=='single').mean()*100:.2f}%")
c3.metric("Avg Song Duration", f"{filtered_df['duration_min'].mean():.2f} min")
c4.metric("Avg Popularity", f"{filtered_df['popularity'].mean():.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Clean Content Share", f"{(~filtered_df['is_explicit']).mean()*100:.2f}%")
c6.metric("Album Share", f"{(filtered_df['album_type']=='album').mean()*100:.2f}%")
c7.metric("Average Rank", f"{filtered_df['position'].mean():.2f}")
c8.metric("Acceptance Score", f"{filtered_df['content_acceptance_score'].mean():.2f}")

st.divider()

st.subheader("1. Explicit vs Clean Content")
left, right = st.columns(2)

with left:
    counts = filtered_df["is_explicit"].map({True: "Explicit", False: "Clean"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Explicit vs Clean Track Count")
    ax.set_xlabel("Content Type")
    ax.set_ylabel("Tracks")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    pop = filtered_df.groupby("is_explicit")["popularity"].mean().rename(index={False: "Clean", True: "Explicit"})
    fig, ax = plt.subplots(figsize=(6, 4))
    pop.plot(kind="bar", ax=ax)
    ax.set_title("Average Popularity: Explicit vs Clean")
    ax.set_xlabel("Content Type")
    ax.set_ylabel("Average Popularity")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("2. Single vs Album Analysis")
left, right = st.columns(2)

with left:
    format_counts = filtered_df["album_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    format_counts.plot(kind="bar", ax=ax)
    ax.set_title("Release Format Distribution")
    ax.set_xlabel("Album Type")
    ax.set_ylabel("Tracks")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

with right:
    format_pop = filtered_df.groupby("album_type")["popularity"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    format_pop.plot(kind="bar", ax=ax)
    ax.set_title("Average Popularity by Release Format")
    ax.set_xlabel("Album Type")
    ax.set_ylabel("Average Popularity")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. Song Duration Analysis")
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(filtered_df["duration_min"].dropna(), bins=30)
    ax.set_title("Song Duration Distribution")
    ax.set_xlabel("Duration (Minutes)")
    ax.set_ylabel("Tracks")
    st.pyplot(fig)
    plt.close(fig)

with right:
    duration_pop = filtered_df.groupby("duration_bucket", observed=True)["popularity"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    duration_pop.plot(kind="bar", ax=ax)
    ax.set_title("Average Popularity by Duration")
    ax.set_xlabel("Duration Group")
    ax.set_ylabel("Average Popularity")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("4. Album Size Analysis")
album_df = filtered_df[filtered_df["album_type"] == "album"].copy()

if not album_df.empty:
    corr = album_df[["total_tracks", "popularity"]].corr().iloc[0, 1]
    st.write(f"**Correlation between album size and popularity:** {corr:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(album_df["total_tracks"], album_df["popularity"], alpha=0.35)
    ax.set_title("Album Size vs Track Popularity")
    ax.set_xlabel("Total Tracks in Album")
    ax.set_ylabel("Popularity")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("5. Rank-Tier Analysis")
tier_analysis = (
    filtered_df.groupby("rank_tier", observed=True)
    .agg(
        tracks=("song", "count"),
        explicit_share=("is_explicit", "mean"),
        avg_popularity=("popularity", "mean"),
        avg_duration=("duration_min", "mean"),
        avg_album_size=("total_tracks", "mean")
    )
    .reset_index()
)

tier_analysis["explicit_share"] *= 100
tier_analysis = tier_analysis.rename(columns={"explicit_share": "explicit_share_pct"})
st.dataframe(tier_analysis, use_container_width=True)

st.subheader("Filtered Dataset")
st.write(f"Showing {len(filtered_df):,} rows after filters.")
st.dataframe(filtered_df, use_container_width=True)

st.subheader("Recommendations")
st.markdown(
    "- Monitor explicit content rather than automatically excluding it.\n"
    "- Maintain clean alternatives where available.\n"
    "- Evaluate singles and album tracks separately.\n"
    "- Avoid using album size alone as a success strategy.\n"
    "- Consider song duration during playlist pitching.\n"
    "- Use rank-tier-specific strategies for Top 10, Top 25 and Top 50."
)

st.info(
    "This dashboard measures playlist representation and metadata-based performance. "
    "It does not represent complete cultural, regulatory, streaming, revenue, or listener-sentiment analysis."
)
