"""
43. Restaurant Revenue Prediction  (v2 - with animated landing page)
Problem Type : Regression
Dataset      : Kaggle - Restaurant Revenue Dataset
Objective    : Predict restaurant revenue using operational data
ML Technique : Random Forest Regressor
Metrics      : RMSE, R2

Run with:  streamlit run restaurant_revenue_app_v2.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Restaurant Revenue Prediction", page_icon="🍽️", layout="wide")

if "page" not in st.session_state:
    st.session_state["page"] = "landing"


# ----------------------------------------------------------------------------
# Landing page
# ----------------------------------------------------------------------------
LANDING_CSS = """
<style>
/* hide sidebar + header chrome on the landing page */
[data-testid="stSidebar"], [data-testid="collapsedControl"], header { display: none !important; }

/* animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0e1117, #1a0b0f, #2a0d12, #0e1117);
    background-size: 400% 400%;
    animation: gradientShift 14s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* floating food emojis */
.floaters { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.floaters span {
    position: absolute; bottom: -80px; font-size: 40px; opacity: 0;
    animation: floatUp linear infinite;
}
@keyframes floatUp {
    0%   { transform: translateY(0) rotate(0deg);       opacity: 0; }
    10%  { opacity: .35; }
    90%  { opacity: .35; }
    100% { transform: translateY(-120vh) rotate(360deg); opacity: 0; }
}

/* hero text */
.hero { position: relative; z-index: 2; text-align: center; padding-top: 6vh; }
.hero h1 {
    font-size: clamp(2.2rem, 6vw, 4.2rem); font-weight: 800; color: #fff;
    margin: 0; animation: fadeUp 1s ease both;
}
.hero h1 .accent {
    background: linear-gradient(90deg, #ff4b4b, #ff9a9a, #ff4b4b);
    background-size: 200% auto; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; animation: shine 4s linear infinite;
}
@keyframes shine { to { background-position: 200% center; } }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* typewriter subtitle */
.typewriter {
    display: inline-block; max-width: 100%; overflow: hidden; white-space: nowrap;
    border-right: 3px solid #ff4b4b; color: #d0d0d0; font-size: clamp(1rem, 2.2vw, 1.5rem);
    margin-top: 14px; width: 0;
    animation: typing 3s steps(46) .8s forwards, caret .7s step-end infinite;
}
@keyframes typing { to { width: 46ch; } }
@keyframes caret  { 50% { border-color: transparent; } }

/* feature cards */
.cards { position: relative; z-index: 2; display: flex; flex-wrap: wrap;
         gap: 18px; justify-content: center; margin: 44px auto 10px; max-width: 980px; }
.card {
    flex: 1 1 210px; max-width: 230px; padding: 22px 18px; text-align: center; color: #fff;
    background: rgba(255,75,75,.06); backdrop-filter: blur(8px);
    border: 1px solid rgba(255,75,75,.30); border-radius: 18px;
    opacity: 0; animation: fadeUp .8s ease forwards; transition: transform .25s, box-shadow .25s;
}
.card:hover { transform: translateY(-8px) scale(1.03); border-color: #ff4b4b;
             box-shadow: 0 0 28px rgba(255,75,75,.35); }
.card .icon { font-size: 2.3rem; display: block; margin-bottom: 8px; animation: bob 3s ease-in-out infinite; }
.card h4 { margin: 0 0 6px; font-size: 1.05rem; color: #fff; }
.card p  { margin: 0; font-size: .88rem; color: #bbb; }
.card:nth-child(1) { animation-delay: 1.6s; }
.card:nth-child(2) { animation-delay: 1.9s; }
.card:nth-child(3) { animation-delay: 2.2s; }
.card:nth-child(4) { animation-delay: 2.5s; }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }

/* call-to-action button */
.stButton > button {
    position: relative; z-index: 3; display: block; margin: 0 auto;
    background: linear-gradient(90deg, #ff4b4b, #b71c1c); color: #fff; border: none;
    padding: 14px 42px; font-size: 1.15rem; font-weight: 700; border-radius: 50px;
    animation: pulse 2s infinite; transition: transform .2s;
}
.stButton > button:hover { transform: scale(1.08); color: #fff; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,75,75,.7); }
    70%  { box-shadow: 0 0 0 22px rgba(255,75,75,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,75,75,0); }
}
</style>
"""

# (emoji, left %, duration s, delay s)
FLOATERS = [
    ("🍕", 5, 14, 0), ("🍔", 15, 18, 3), ("🍜", 25, 16, 6), ("🍣", 35, 20, 1),
    ("🌮", 45, 15, 8), ("🥗", 55, 19, 4), ("🍩", 65, 17, 10), ("☕", 75, 21, 2),
    ("🍝", 85, 16, 7), ("🥘", 93, 18, 5), ("💰", 20, 22, 12), ("📈", 70, 23, 9),
]

STATS_HTML = """
<div style="display:flex;justify-content:center;gap:48px;flex-wrap:wrap;
            font-family:sans-serif;color:#fff;text-align:center;padding:6px 0;">
  <div><div style="font-size:2.4rem;font-weight:800;color:#ff4b4b" data-to="200" data-suffix="">0</div>
       <div style="opacity:.75;color:#ccc">Decision trees</div></div>
  <div><div style="font-size:2.4rem;font-weight:800;color:#ff4b4b" data-to="2" data-suffix="">0</div>
       <div style="opacity:.75;color:#ccc">Evaluation metrics</div></div>
  <div><div style="font-size:2.4rem;font-weight:800;color:#ff4b4b" data-to="100" data-suffix="%">0</div>
       <div style="opacity:.75;color:#ccc">Interactive</div></div>
</div>
<script>
document.querySelectorAll('[data-to]').forEach(el => {
  const target = +el.dataset.to, suffix = el.dataset.suffix || '';
  const start = performance.now() + 2600, dur = 1800;
  function tick(now) {
    if (now < start) return requestAnimationFrame(tick);
    const p = Math.min((now - start) / dur, 1);
    el.textContent = Math.floor(target * (1 - Math.pow(1 - p, 3))) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
</script>
"""


def landing_page() -> None:
    st.markdown(LANDING_CSS, unsafe_allow_html=True)

    floaters = "".join(
        f'<span style="left:{left}%;animation-duration:{dur}s;animation-delay:{delay}s">{emoji}</span>'
        for emoji, left, dur, delay in FLOATERS
    )
    st.markdown(f'<div class="floaters">{floaters}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>🍽️ Restaurant <span class="accent">Revenue</span> Prediction</h1>
            <div class="typewriter">Forecast restaurant revenue with machine learning.</div>
        </div>
        <div class="cards">
            <div class="card"><span class="icon">📊</span><h4>Explore Data</h4>
                <p>Upload the Kaggle dataset and inspect it instantly.</p></div>
            <div class="card"><span class="icon">🌲</span><h4>Random Forest</h4>
                <p>Train a regressor on location, seating and promotions.</p></div>
            <div class="card"><span class="icon">🎯</span><h4>RMSE &amp; R²</h4>
                <p>See exactly how accurate the predictions are.</p></div>
            <div class="card"><span class="icon">🔮</span><h4>Predict</h4>
                <p>Estimate revenue for any new restaurant.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    components.html(STATS_HTML, height=110)

    if st.button("🚀 Get Started"):
        st.session_state["page"] = "app"
        st.rerun()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def make_demo_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic data so the app works without the Kaggle file."""
    rng = np.random.default_rng(seed)
    location = rng.choice(["Downtown", "Suburb", "Mall", "Highway", "Airport"], n)
    cuisine = rng.choice(["Indian", "Italian", "Chinese", "Mexican", "Fast Food"], n)
    seating = rng.integers(20, 200, n)
    promotions = rng.integers(0, 2, n)
    marketing = rng.uniform(1000, 20000, n).round(2)
    rating = rng.uniform(2.5, 5.0, n).round(1)
    price = rng.uniform(5, 60, n).round(2)

    loc_boost = pd.Series(location).map(
        {"Downtown": 1.3, "Mall": 1.2, "Airport": 1.4, "Suburb": 1.0, "Highway": 0.9}
    ).values
    revenue = (
        seating * price * 6 * loc_boost
        + promotions * 15000
        + marketing * 2.5
        + rating * 8000
        + rng.normal(0, 15000, n)
    ).round(2)

    return pd.DataFrame({
        "Location": location,
        "Cuisine": cuisine,
        "Seating Capacity": seating,
        "Promotions": promotions,
        "Marketing Budget": marketing,
        "Rating": rating,
        "Average Meal Price": price,
        "Revenue": revenue,
    })


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame, n_estimators: int, max_depth: int | None, seed: int) -> Pipeline:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("rf", model)])


def feature_importances(pipe: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Aggregate one-hot importances back to the original column names."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=names)
    agg = {}
    for name, val in imp.items():
        base = name.split("__", 1)[1]
        for col in X.columns:
            if base == col or base.startswith(col + "_"):
                base = col
                break
        agg[base] = agg.get(base, 0) + val
    return pd.Series(agg).sort_values(ascending=True)


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
def main_app() -> None:
    st.title("🍽️ Restaurant Revenue Prediction")
    st.caption(
        "Regression project using a Random Forest Regressor on operational data "
        "(location, seating capacity, promotions, etc.). Metrics: RMSE and R²."
    )

    if st.sidebar.button("🏠 Back to home"):
        st.session_state["page"] = "landing"
        st.rerun()

    # ---- Sidebar: data + hyperparameters ----
    st.sidebar.header("1. Data")
    uploaded = st.sidebar.file_uploader("Upload Kaggle Restaurant Revenue CSV", type=["csv"])

    if uploaded is not None:
        df = load_csv(uploaded)
        st.sidebar.success("Using uploaded dataset")
    else:
        df = make_demo_data()
        st.sidebar.info("No file uploaded - using synthetic demo data")

    df = df.dropna(axis=1, how="all")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("The dataset needs at least one numeric column to use as the revenue target.")
        st.stop()

    default_target = "Revenue" if "Revenue" in df.columns else numeric_cols[-1]
    target = st.sidebar.selectbox("Target column", numeric_cols, index=numeric_cols.index(default_target))

    candidate_features = [c for c in df.columns if c != target]
    default_features = [
        c for c in candidate_features
        if not (df[c].dtype == "object" and df[c].nunique() > 30)
    ]
    features = st.sidebar.multiselect("Input features", candidate_features, default=default_features)

    st.sidebar.header("2. Model")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.sidebar.slider("Number of trees", 50, 500, 200, 50)
    max_depth_opt = st.sidebar.slider("Max depth (0 = unlimited)", 0, 30, 0)
    max_depth = None if max_depth_opt == 0 else max_depth_opt
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    if not features:
        st.warning("Select at least one input feature in the sidebar.")
        st.stop()

    tab_data, tab_train, tab_predict = st.tabs(["📊 Data", "🤖 Train & Evaluate", "🔮 Predict"])

    with tab_data:
        st.subheader("Dataset preview")
        st.dataframe(df.head(50), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))

        st.subheader("Summary statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.subheader(f"Distribution of {target}")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df[target].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with tab_train:
        st.subheader("Train the Random Forest model")
        if st.button("🚀 Train model", type="primary"):
            data = df[features + [target]].dropna()
            X, y = data[features], data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(seed)
            )
            pipe = build_pipeline(X, n_estimators, max_depth, int(seed))
            with st.spinner("Training..."):
                pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))

            st.session_state["pipe"] = pipe
            st.session_state["features"] = features
            st.session_state["train_df"] = X
            st.session_state["metrics"] = (rmse, r2)
            st.session_state["results"] = (y_test.values, preds)
            st.session_state["importance"] = feature_importances(pipe, X)

        if "metrics" in st.session_state:
            rmse, r2 = st.session_state["metrics"]
            m1, m2 = st.columns(2)
            m1.metric("RMSE (lower is better)", f"{rmse:,.2f}")
            m2.metric("R² (closer to 1 is better)", f"{r2:.3f}")

            y_true, y_pred = st.session_state["results"]
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Actual vs Predicted**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_true, y_pred, alpha=0.5, color="#55A868")
                lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
                ax.plot(lims, lims, "r--", label="Perfect prediction")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.legend()
                st.pyplot(fig)

            with col_b:
                st.markdown("**Feature importance**")
                imp = st.session_state["importance"]
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.barh(imp.index, imp.values, color="#C44E52")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
        else:
            st.info("Click **Train model** to fit the Random Forest and see RMSE / R².")

    with tab_predict:
        st.subheader("Predict revenue for a new restaurant")
        if "pipe" not in st.session_state:
            st.info("Train the model first in the **Train & Evaluate** tab.")
        else:
            X_ref = st.session_state["train_df"]
            feats = st.session_state["features"]
            inputs = {}
            cols = st.columns(3)
            for i, f in enumerate(feats):
                with cols[i % 3]:
                    s = X_ref[f]
                    if s.dtype == "object" or str(s.dtype) in ("category", "bool"):
                        inputs[f] = st.selectbox(f, sorted(s.astype(str).unique()))
                    elif pd.api.types.is_integer_dtype(s):
                        inputs[f] = st.number_input(
                            f, min_value=int(s.min()), max_value=int(s.max()), value=int(s.median())
                        )
                    else:
                        inputs[f] = st.number_input(
                            f, min_value=float(s.min()), max_value=float(s.max()),
                            value=float(s.median())
                        )

            if st.button("Predict revenue"):
                row = pd.DataFrame([inputs])
                pred = st.session_state["pipe"].predict(row)[0]
                st.success(f"Estimated revenue: **{pred:,.2f}**")
                st.caption("Outcome: business performance forecasting.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
if st.session_state["page"] == "landing":
    landing_page()
else:
    main_app()