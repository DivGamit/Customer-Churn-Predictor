import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_on_uploaded_data(df):
    # Auto-find churn column
    target_cols = [c for c in df.columns if "churn" in c.lower() or "leave" in c.lower() or "exit" in c.lower()]
    if not target_cols:
        st.error("No churn column found. Name it something like 'Churn', 'WillLeave', etc.")
        return None
    target = target_cols[0]

    # Clean target column
    def clean_churn(val):
     if pd.isna(val):
        return None
     s = str(val).strip().lower()
     return 1 if s in ['yes','1','true','churn','leave','1.0'] else 0

    df[target] = df[target].apply(clean_churn)
    df = df.dropna(subset=[target])

    # Auto-encode all features with dummies (industry standard)
    X_encoded = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]

    # Train model
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_encoded, y)

    return model, X_encoded.columns, target


st.title("Universal Customer Churn Predictor")
st.markdown("**Upload any CSV → Get instant churn predictions**") 

uploaded_file = st.file_uploader("Drop your customer CSV here", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df):,} rows")
    st.dataframe(df.head())

    if st.button("Train Model & Predict Churn", type="primary"):
        with st.spinner("Training model on your data (takes 3–15 seconds)..."):
            result = train_on_uploaded_data(df.copy())
            if result is None:
                st.stop()

            model, feature_cols, target = result

            # Predict on the same data
            X_new = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
            # Fix missing/extra columns
            for col in feature_cols:
                if col not in X_new.columns:
                    X_new[col] = 0
            X_new = X_new[feature_cols]

            df["Churn_Probability"] = model.predict_proba(X_new)[:, 1]
            df["Prediction"] = df["Churn_Probability"].apply(
                lambda x: "Will Churn" if x > 0.5 else "Will Stay"
            )

        st.success("Predictions ready!")
        st.balloons()

        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Customers", len(df))
            st.metric("Predicted to Churn", len(df[df["Prediction"] == "Will Churn"]))
        with col2:
            st.metric("Average Churn Risk", f"{df['Churn_Probability'].mean():.1%}")

        st.subheader("Top 10 Riskiest Customers")
        st.dataframe(
            df.sort_values("Churn_Probability", ascending=False)
              .head(10)
              .style.background_gradient(cmap="Reds", subset=["Churn_Probability"])
        )

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download Full Predictions CSV",
            csv,
            "churn_predictions.csv",
            "text/csv",
            use_container_width=True
        )