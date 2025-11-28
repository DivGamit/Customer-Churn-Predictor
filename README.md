# Customer-Churn-Predictor
Customer Churn Predictor – Streamlit | Python | Pandas | Scikit-learn  
Built a production-ready web app that accepts any customer CSV, auto-cleans & encodes features, trains Random Forest model on-the-fly, and returns churn predictions with downloadable results. Deployed on Streamlit Cloud. Used by recruiters in live interviews.

- Upload any customer CSV (any columns, any format)  
- Auto-detects churn column & cleans Yes/No/1/0/True automatically  
- Trains Random Forest model  
- Shows churn probability + "Will Churn / Will Stay" for every customer  
- Download full predictions CSV  
- no data stored, each user gets their own model  

# Tech Stack
- Python · Streamlit · Pandas · Scikit-learn  
- Production-grade: `pd.get_dummies(drop_first=True)` + feature alignment  
- No pre-trained model — trains live on user data
