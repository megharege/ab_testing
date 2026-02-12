import streamlit as st
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(layout="wide")
st.title("🎯 Reinforcement Learning — UCB A/B Testing Optimizer")

st.markdown(
    """
    This app uses the **Upper Confidence Bound (UCB)** algorithm to optimize
    A/B testing by dynamically selecting the best-performing variant.
    """
)

# ==========================================
# DATASET FORMAT INFO
# ==========================================
st.subheader("📂 Dataset Format")

st.markdown(
    """
    Please upload a **CSV file** with the following structure:

    - Each **column** = one Ad / Variant (Ad1, Ad2, Ad3, ...)
    - Each **row** = one user interaction (round)
    - Values must be **binary (0 or 1)**  
        - `1` → Click / Success  
        - `0` → No Click / Failure
    """
)

example_df = pd.DataFrame({
    "Ad1": [1, 0, 0, 1, 0],
    "Ad2": [0, 1, 0, 0, 1],
    "Ad3": [0, 0, 1, 0, 0]
})

st.markdown("**Example CSV format:**")
st.dataframe(example_df)

example_csv = example_df.to_csv(index=False).encode()
st.download_button(
    "⬇ Download Sample Dataset",
    example_csv,
    "sample_ab_testing_dataset.csv",
    mime="text/csv"
)

st.divider()

# ==========================================
# UCB FUNCTION
# ==========================================
def run_ucb(dataset):

    N = dataset.shape[0]      # rounds
    d = dataset.shape[1]      # number of ads

    ads_selected = []
    numbers_of_selections = [0] * d
    sums_of_rewards = [0] * d
    total_reward = 0

    for n in range(N):

        ad = 0
        max_upper_bound = 0

        for i in range(d):

            if numbers_of_selections[i] > 0:
                avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                upper_bound = avg_reward + delta
            else:
                upper_bound = 1e400

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i

        ads_selected.append(ad)
        numbers_of_selections[ad] += 1

        reward = dataset.iloc[n, ad]
        sums_of_rewards[ad] += reward
        total_reward += reward

    return ads_selected, numbers_of_selections, sums_of_rewards, total_reward


# ==========================================
# FILE UPLOAD
# ==========================================
file = st.file_uploader(
    "📤 Upload A/B Testing Dataset (CSV)",
    type=["csv"]
)

if file:

    df = pd.read_csv(file)

    # --------------------------------------
    # INPUT VALIDATION
    # --------------------------------------
    if df.isnull().values.any():
        st.error("Dataset contains missing values. Please clean your data.")
        st.stop()

    if not set(df.values.flatten()).issubset({0, 1}):
        st.error("Dataset must contain only 0 and 1 values.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    N, d = df.shape
    st.info(f"👥 Rounds (users): {N} | 🧪 Variants (ads): {d}")

    # ======================================
    # RUN UCB
    # ======================================
    if st.button("🚀 Run UCB Optimization"):

        progress = st.progress(0)
        start = time.time()

        ads_selected, selections, rewards, total_reward = run_ucb(df)

        end = time.time()
        progress.progress(100)

        st.success("Optimization Completed ✅")

        # ----------------------------------
        # RESULTS
        # ----------------------------------
        st.subheader("📊 Results")

        best_ad = np.argmax(rewards)

        st.write(f"🏆 **Best Ad:** Ad{best_ad + 1}")
        st.write(f"🎁 **Total Reward:** {total_reward}")
        st.write(f"⏱ **Time Taken:** {round(end - start, 2)} seconds")

        ctrs = [r / s if s > 0 else 0 for r, s in zip(rewards, selections)]

        result_df = pd.DataFrame({
            "Ad": [f"Ad{i+1}" for i in range(d)],
            "Selections": selections,
            "Rewards": rewards,
            "CTR": ctrs
        })

        st.dataframe(result_df)

        # ----------------------------------
        # HISTOGRAM
        # ----------------------------------
        st.subheader("📈 Ad Selection Frequency")

        fig, ax = plt.subplots()
        ax.hist(ads_selected, bins=d)
        ax.set_xlabel("Ad Index")
        ax.set_ylabel("Times Selected")
        ax.set_title("Histogram of Ad Selections")

        st.pyplot(fig)

        # ----------------------------------
        # DOWNLOAD RESULTS
        # ----------------------------------
        summary_csv = result_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Summary (per Ad)",
            summary_csv,
            "ucb_summary.csv",
            mime="text/csv"
        )

        history_df = pd.DataFrame({
            "Round": range(1, len(ads_selected) + 1),
            "Ad_Selected": [f"Ad{i+1}" for i in ads_selected],
            "Reward": [df.iloc[i, ads_selected[i]] for i in range(len(ads_selected))]
        })

        history_csv = history_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Full History (per Round)",
            history_csv,
            "ucb_history.csv",
            mime="text/csv"
        )
