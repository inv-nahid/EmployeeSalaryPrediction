import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="💰 Salary Predictor", layout="wide")

# Load model and data


@st.cache_resource
def load_model():
    try:
        return joblib.load("salary_predictor_model_enhanced.joblib")
    except:
        st.error("Model file not found!")
        return None


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Salary Data.csv")
        return df.dropna(subset=["Salary"])
    except:
        # Generate sample data
        np.random.seed(42)
        n = 500
        data = {
            'Age': np.random.randint(22, 65, n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'Education Level': np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], n),
            'Job Title': np.random.choice(['Engineer', 'Manager', 'Analyst', 'Developer'], n),
            'Years of Experience': np.random.randint(0, 20, n),
        }
        data['Salary'] = (40000 + data['Years of Experience'] * 2000 +
                          np.random.normal(0, 10000, n)).astype(int)
        return pd.DataFrame(data)


model = load_model()
df = load_data()

# Title
st.title("💰 Salary Prediction App")

# Sidebar for prediction
st.sidebar.header("🔮 Predict Your Salary")

age = st.sidebar.slider("Age", 18, 70, 30)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox(
    "Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
job_title = st.sidebar.selectbox(
    "Job Title", ["Engineer", "Manager", "Analyst", "Developer"])

if st.sidebar.button("Predict Salary"):
    if model:
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [job_title],
            'Years of Experience': [experience]
        })

        prediction = model.predict(input_data)[0]
        st.sidebar.success(f"Predicted Salary: ${prediction:,.0f}")

# Main content tabs
tab1, tab2, tab3 = st.tabs(
    ["📊 Data Overview", "📈 Analysis", "🔍 Model Performance"])

with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Average Salary", f"${df['Salary'].mean():,.0f}")
    col3.metric("Salary Range",
                f"${df['Salary'].max()-df['Salary'].min():,.0f}")
    col4.metric("Features", len(df.columns)-1)

    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    st.subheader("Salary Distribution")
    fig = px.histogram(df, x='Salary', nbins=30, title="Salary Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salary by Education")
        fig = px.box(df, x='Education Level', y='Salary')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Salary by Experience")
        fig = px.scatter(df, x='Years of Experience',
                         y='Salary', color='Gender')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Salary by Job Title")
    job_avg = df.groupby('Job Title')[
        'Salary'].mean().sort_values(ascending=False)
    fig = px.bar(x=job_avg.index, y=job_avg.values)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if model:
        st.header("Model Performance")

        X = df.drop('Salary', axis=1)
        y = df['Salary']
        predictions = model.predict(X)

        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("RMSE", f"${rmse:,.0f}")

        st.subheader("Predicted vs Actual")
        fig = px.scatter(x=y, y=predictions,
                         labels={'x': 'Actual Salary', 'y': 'Predicted Salary'})
        # Add perfect prediction line
        min_val, max_val = y.min(), y.max()
        fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='Perfect Prediction',
                        line=dict(dash='dash', color='red'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model not available for performance analysis")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Machine Learning Model: Random Forest")
