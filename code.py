import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("📊 Data Cleaning + Analysis + ML App")

# -------------------------
# MYSQL CONFIG
# -------------------------
host = "localhost"
user = "root"
password = quote_plus("REKAMITHRA@1997")
database = "Employee_Details"

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

    df = st.session_state.df

    # -------------------------
    # RAW DATA
    # -------------------------
    st.subheader("📊 Raw Data")
    st.dataframe(df, use_container_width=True)

    # -------------------------
    # DATA ANALYSIS
    # -------------------------
    st.subheader("🔍 Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Null Values")
        st.write(df.isnull().sum())

    with col2:
        st.write("Duplicate Rows")
        st.write(df.duplicated().sum())

    # -------------------------
    # DATA CLEANING
    # -------------------------
    st.subheader("🧹 Data Cleaning")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Remove Duplicates"):
            st.session_state.df = st.session_state.df.drop_duplicates()
            st.success("✅ Duplicates Removed")

    with col2:
        option = st.selectbox(
            "Handle Null Values",
            ["None", "Drop Null Rows", "Fill Mean", "Fill Median", "Fill Mode", "Fill Custom"]
        )

        if option != "None":
            df_temp = st.session_state.df.copy()

            if option == "Drop Null Rows":
                df_temp = df_temp.dropna()

            elif option == "Fill Mean":
                for col in df_temp.select_dtypes(include=np.number):
                    df_temp[col] = df_temp[col].fillna(df_temp[col].mean())

            elif option == "Fill Median":
                for col in df_temp.select_dtypes(include=np.number):
                    df_temp[col] = df_temp[col].fillna(df_temp[col].median())

            elif option == "Fill Mode":
                df_temp = df_temp.fillna(df_temp.mode().iloc[0])

            elif option == "Fill Custom":
                value = st.text_input("Enter Value")
                if value:
                    df_temp = df_temp.fillna(value)

            st.session_state.df = df_temp
            st.success("✅ Null Values Processed")

    df = st.session_state.df

    # -------------------------
    # CLEANED DATA
    # -------------------------
    st.subheader("🧹 Cleaned Data")
    st.dataframe(df, use_container_width=True)

    # -------------------------
    # DASHBOARD
    # -------------------------
    st.subheader("📈 Interactive Dashboard")

    all_columns = df.columns.tolist()

    x_col = st.selectbox("Select X-axis", all_columns)
    y_col = st.selectbox("Select Y-axis", all_columns)

    chart_type = st.selectbox(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"]
    )

    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_col, y=y_col)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col)

    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col)

    elif chart_type == "Pie Chart":
        pie_data = df.groupby(x_col)[y_col].sum().reset_index()
        fig = px.pie(pie_data, names=x_col, values=y_col)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # 🤖 MACHINE LEARNING
    # -------------------------
    st.subheader("🤖 Machine Learning (Regression)")

    df_ml = df.copy()

    # Convert categorical → numeric
    df_ml = pd.get_dummies(df_ml, drop_first=True)

    numeric_cols = df_ml.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:

        feature = st.selectbox("Select Feature (X)", df.columns)
        target = st.selectbox("Select Target (Y)", numeric_cols)

        if st.button("Train Model"):

            if feature == target:
                st.error("❌ Feature and Target should be different")
            else:
                X = df_ml[["Employee_ID", "Employee_Name", "Age", "Department", "Experience"]]
                y = df_ml["Salary"]

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Score
                score = r2_score(y_test, y_pred)

                st.success(f"✅ Model Trained | R² Score: {score:.2f}")

                # Sort values for smooth line
                sorted_idx = X_test[feature].argsort()
                X_sorted = X_test.iloc[sorted_idx]
                y_sorted = y_test.iloc[sorted_idx]
                y_pred_sorted = y_pred[sorted_idx]

                # Plot
                fig_ml = go.Figure()

                fig_ml.add_trace(go.Scatter(
                    x=X_sorted[feature],
                    y=y_sorted,
                    mode='markers',
                    name='Actual'
                ))

                fig_ml.add_trace(go.Scatter(
                    x=X_sorted[feature],
                    y=y_pred_sorted,
                    mode='lines',
                    name='Regression Line'
                ))

                fig_ml.update_layout(
                    title="📈 Regression Line",
                    xaxis_title=feature,
                    yaxis_title=target
                )

                st.plotly_chart(fig_ml, use_container_width=True)

    else:
        st.warning("⚠ Need at least 2 numeric columns")

    # -------------------------
    # EXPORT
    # -------------------------
    st.subheader("📤 Export Excel")

    output = BytesIO()
    df.to_excel(output, index=False)

    st.download_button(
        label="Download Excel",
        data=output.getvalue(),
        file_name="cleaned_data.xlsx"
    )

    # -------------------------
    # MYSQL UPLOAD
    # -------------------------
    st.subheader("📦 Upload Data to MySQL")

    if st.button("Upload to MySQL"):
        try:
            engine = create_engine(
                f"mysql+pymysql://{user}:{password}@{host}/{database}"
            )

            df.to_sql(
                "cleaned_employees",
                engine,
                if_exists="replace",
                index=False
            )

            st.success("✅ Uploaded to MySQL Successfully")

        except Exception as e:
            st.error(f"❌ Upload Failed: {e}")
