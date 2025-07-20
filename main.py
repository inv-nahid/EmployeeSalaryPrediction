import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Salary Prediction Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data


@st.cache_resource
def load_model():
    try:
        model = joblib.load("salary_predictor_model_enhanced.joblib")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run the training script first.")
        return None


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Salary Data.csv")
        df.dropna(subset=["Salary"], inplace=True)
        return df
    except FileNotFoundError:
        # Create sample data if file not found
        np.random.seed(42)
        n_samples = 1000

        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        education_levels = np.random.choice(
            ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'],
            n_samples, p=[0.2, 0.4, 0.3, 0.1]
        )
        job_titles = np.random.choice([
            'Software Engineer', 'Data Scientist', 'Manager', 'Analyst',
            'Developer', 'Consultant', 'Director', 'Senior Engineer'
        ], n_samples)
        ages = np.random.randint(22, 65, n_samples)
        # Generate experience based on age constraints
        experience = np.array(
            [np.random.randint(0, min(age-22, 40) + 1) for age in ages])

        # Create salary based on factors with more realistic calculations
        base_salary = 40000
        education_bonus = np.where(education_levels == 'PhD', 25000,
                                   np.where(education_levels == 'Master\'s', 15000,
                                            np.where(education_levels == 'Bachelor\'s', 8000, 0)))

        salary = (base_salary +
                  experience * 2500 +
                  (ages - 22) * 800 +
                  education_bonus +
                  np.random.normal(0, 8000, n_samples))

        # Ensure minimum salary and realistic ranges
        salary = np.maximum(salary, 35000)
        salary = np.minimum(salary, 200000)  # Cap at reasonable maximum

        df = pd.DataFrame({
            'Age': ages,
            'Gender': genders,
            'Education Level': education_levels,
            'Job Title': job_titles,
            'Years of Experience': experience,
            'Salary': salary
        })
        return df


# Load resources
model = load_model()
df = load_data()

# Title and Header
st.markdown('<h1 class="main-header">💰 Salary Prediction Analytics Dashboard</h1>',
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home & Overview", "🔮 Salary Predictor", "📊 Data Analysis",
        "📈 Model Performance", "🔍 Insights & Trends"]
)

if page == "🏠 Home & Overview":
    st.header("📋 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
    with col3:
        st.metric("Salary Range",
                  f"${df['Salary'].max() - df['Salary'].min():,.0f}")
    with col4:
        st.metric("Features", len(df.columns) - 1)

    st.markdown("---")

    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Salary Distribution")
        fig = px.histogram(
            df, x='Salary', nbins=30,
            title="Salary Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Quick Stats")
        stats_df = df['Salary'].describe().round(2)
        for stat, value in stats_df.items():
            if stat in ['mean', 'std', 'min', 'max']:
                st.metric(stat.title(), f"${value:,.0f}")

elif page == "🔮 Salary Predictor":
    st.header("🔮 Predict Your Salary")

    if model is None:
        st.error("Model not available. Please check the model file.")
        st.stop()

    st.markdown("Fill in your details below to get a salary prediction:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 30)
        experience = st.slider("Years of Experience", 0, 50, 5)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's", "PhD"]
        )

    job_title = st.selectbox(
        "Job Title",
        ["Software Engineer", "Data Scientist", "Manager", "Analyst",
         "Developer", "Consultant", "Director", "Senior Engineer"]
    )

    if st.button("🔮 Predict Salary", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [job_title],
            'Years of Experience': [experience]
        })

        try:
            prediction = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="prediction-box">
                Predicted Salary: ${prediction:,.0f}
            </div>
            """, unsafe_allow_html=True)

            # Confidence interval simulation
            std_salary = df['Salary'].std()
            lower_bound = max(prediction - std_salary * 0.2, 0)
            upper_bound = prediction + std_salary * 0.2

            st.info(
                f"💡 **Confidence Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")

            # Comparison with similar profiles
            similar_profiles = df[
                (df['Education Level'] == education) &
                (df['Gender'] == gender) &
                (abs(df['Age'] - age) <= 5) &
                (abs(df['Years of Experience'] - experience) <= 3)
            ]

            if len(similar_profiles) > 0:
                avg_similar = similar_profiles['Salary'].mean()
                percentile = (
                    similar_profiles['Salary'] < prediction).mean() * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similar Profiles Avg", f"${avg_similar:,.0f}")
                with col2:
                    st.metric("Your Percentile", f"{percentile:.0f}%")
                with col3:
                    st.metric("Sample Size", len(similar_profiles))

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

elif page == "📊 Data Analysis":
    st.header("📊 Comprehensive Data Analysis")

    # Correlation Analysis
    st.subheader("🔗 Feature Correlations")

    # Prepare numerical data for correlation
    df_corr = df.copy()
    df_corr = pd.get_dummies(
        df_corr, columns=['Gender', 'Education Level', 'Job Title'])

    # Select only numerical columns for correlation
    numerical_cols = df_corr.select_dtypes(include=[np.number]).columns
    corr_matrix = df_corr[numerical_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Salary Analysis by Categories
    st.subheader("💰 Salary Analysis by Categories")

    tab1, tab2, tab3 = st.tabs(["By Education", "By Gender", "By Job Title"])

    with tab1:
        fig = px.box(
            df, x='Education Level', y='Salary',
            title="Salary Distribution by Education Level",
            color='Education Level'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Average salary by education
        edu_avg = df.groupby('Education Level')[
            'Salary'].agg(['mean', 'count']).round(0)
        st.dataframe(edu_avg, use_container_width=True)

    with tab2:
        fig = px.violin(
            df, x='Gender', y='Salary',
            title="Salary Distribution by Gender",
            color='Gender', box=True
        )
        st.plotly_chart(fig, use_container_width=True)

        gender_stats = df.groupby('Gender')['Salary'].describe().round(0)
        st.dataframe(gender_stats, use_container_width=True)

    with tab3:
        job_avg = df.groupby('Job Title')[
            'Salary'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=job_avg.values, y=job_avg.index,
            title="Average Salary by Job Title",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Experience vs Salary Analysis
    st.subheader("📈 Experience vs Salary Relationship")

    fig = px.scatter(
        df, x='Years of Experience', y='Salary',
        color='Education Level', size='Age',
        title="Salary vs Experience (Size = Age, Color = Education)",
        hover_data=['Gender', 'Job Title']
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Model Performance":
    st.header("📈 Model Performance Analysis")

    if model is None:
        st.error("Model not available for performance analysis.")
        st.stop()

    # Make predictions on the dataset
    X = df.drop('Salary', axis=1)
    y = df['Salary']

    try:
        predictions = model.predict(X)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"${rmse:,.0f}")
        with col3:
            st.metric("MAE", f"${mae:,.0f}")
        with col4:
            st.metric("Accuracy", f"{(1 - mae/y.mean())*100:.1f}%")

        # Prediction vs Actual Plot
        st.subheader("🎯 Predictions vs Actual Values")

        fig = px.scatter(
            x=y, y=predictions,
            labels={'x': 'Actual Salary', 'y': 'Predicted Salary'},
            title="Predicted vs Actual Salary"
        )
        # Add diagonal line
        min_val, max_val = min(y.min(), predictions.min()), max(
            y.max(), predictions.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Residuals Analysis
        st.subheader("📊 Residuals Analysis")
        residuals = y - predictions

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                residuals, nbins=30,
                title="Distribution of Residuals"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                x=predictions, y=residuals,
                labels={'x': 'Predicted Salary', 'y': 'Residuals'},
                title="Residuals vs Predicted Values"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        # Feature Importance (if available)
        try:
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                st.subheader("🔍 Feature Importance")

                # Get feature names
                numerical_features = ['Age', 'Years of Experience']
                categorical_features = model.named_steps['preprocessor'].named_transformers_[
                    'cat'].named_steps['encoder'].get_feature_names_out(['Gender', 'Education Level', 'Job Title'])
                feature_names = numerical_features + list(categorical_features)

                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.named_steps['regressor'].feature_importances_
                }).sort_values('Importance', ascending=True)

                fig = px.bar(
                    importance_df.tail(10), x='Importance', y='Feature',
                    title="Top 10 Feature Importance",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Feature importance analysis not available.")

    except Exception as e:
        st.error(f"Error in model performance analysis: {str(e)}")

elif page == "🔍 Insights & Trends":
    st.header("🔍 Key Insights & Trends")

    # Key insights
    st.markdown("""
    <div class="insight-box">
    <h3>💡 Key Findings from the Data</h3>
    </div>
    """, unsafe_allow_html=True)

    # Calculate insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎓 Education Impact")
        edu_impact = df.groupby('Education Level')[
            'Salary'].mean().sort_values(ascending=False)

        for edu, salary in edu_impact.items():
            st.write(f"• **{edu}**: ${salary:,.0f}")

        st.subheader("⚡ Experience Premium")
        exp_premium = df.groupby(pd.cut(df['Years of Experience'], bins=[
                                 0, 5, 10, 20, 50], labels=['0-5', '6-10', '11-20', '20+']))['Salary'].mean()

        for exp_range, salary in exp_premium.items():
            st.write(f"• **{exp_range} years**: ${salary:,.0f}")

    with col2:
        st.subheader("👔 Top Paying Jobs")
        job_pay = df.groupby('Job Title')['Salary'].mean(
        ).sort_values(ascending=False).head(5)

        for job, salary in job_pay.items():
            st.write(f"• **{job}**: ${salary:,.0f}")

        st.subheader("⚖️ Gender Pay Analysis")
        gender_pay = df.groupby('Gender')['Salary'].mean()

        for gender, salary in gender_pay.items():
            st.write(f"• **{gender}**: ${salary:,.0f}")

    # Trend Analysis
    st.subheader("📊 Interactive Trend Explorer")

    trend_type = st.selectbox(
        "Select trend to explore:",
        ["Salary vs Age", "Salary vs Experience",
            "Education Distribution", "Job Title Distribution"]
    )

    if trend_type == "Salary vs Age":
        fig = px.scatter(
            df, x='Age', y='Salary', color='Gender',
            size='Years of Experience',
            title="Salary Trends by Age and Gender"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif trend_type == "Salary vs Experience":
        fig = px.line(
            df.groupby('Years of Experience')['Salary'].mean().reset_index(),
            x='Years of Experience', y='Salary',
            title="Average Salary by Years of Experience"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif trend_type == "Education Distribution":
        edu_dist = df['Education Level'].value_counts()
        fig = px.pie(
            values=edu_dist.values, names=edu_dist.index,
            title="Education Level Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif trend_type == "Job Title Distribution":
        job_dist = df['Job Title'].value_counts().head(10)
        fig = px.bar(
            x=job_dist.index, y=job_dist.values,
            title="Top 10 Job Titles Distribution"
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Advanced Analysis
    st.subheader("🧠 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Salary Growth Potential**")

        # Calculate salary growth by experience brackets
        df['Experience_Bracket'] = pd.cut(
            df['Years of Experience'],
            bins=[0, 2, 5, 10, 20, 50],
            labels=['Entry (0-2)', 'Junior (2-5)', 'Mid (5-10)',
                    'Senior (10-20)', 'Expert (20+)']
        )

        growth_potential = df.groupby(['Education Level', 'Experience_Bracket'])[
            'Salary'].mean().unstack()

        if not growth_potential.empty:
            fig = px.line(
                growth_potential.T,
                title="Salary Growth by Experience and Education",
                labels={'index': 'Experience Level', 'value': 'Average Salary'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**🎯 Market Positioning**")

        # Salary percentiles
        percentiles = [10, 25, 50, 75, 90]
        salary_percentiles = [
            df['Salary'].quantile(p/100) for p in percentiles]

        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}th" for p in percentiles],
            'Salary': salary_percentiles
        })

        fig = px.bar(
            percentile_df, x='Percentile', y='Salary',
            title="Salary Percentiles in Dataset",
            text='Salary'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Recommendations Section
    st.markdown("""
    <div class="insight-box">
    <h3>🎯 Career Development Recommendations</h3>
    </div>
    """, unsafe_allow_html=True)

    recommendations = [
        "📚 **Education ROI**: Higher education levels show significant salary premiums. Consider advanced degrees for long-term career growth.",
        "⏰ **Experience Matters**: Each year of experience typically adds $2,000-5,000 to salary potential.",
        "💼 **Strategic Job Selection**: Technical roles (Software Engineer, Data Scientist) typically offer higher compensation.",
        "🎯 **Skill Development**: Focus on high-demand skills in your field to increase market value.",
        "📈 **Career Planning**: Plan for 5-10 year career trajectory to maximize earning potential."
    ]

    for rec in recommendations:
        st.markdown(rec)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>💰 Salary Prediction Analytics Dashboard | Built with Streamlit & Machine Learning</p>
    <p>📊 Powered by Random Forest Regression | Data-Driven Career Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional features
with st.sidebar:
    st.markdown("---")
    st.subheader("📋 Quick Stats")

    if len(df) > 0:
        st.metric("Dataset Size", f"{len(df):,}")
        st.metric("Salary Range",
                  f"${df['Salary'].min():,.0f} - ${df['Salary'].max():,.0f}")
        st.metric("Avg Experience",
                  f"{df['Years of Experience'].mean():.1f} years")
        st.metric("Most Common Job", df['Job Title'].mode()[0])

    st.markdown("---")
    st.subheader("🔧 Model Info")
    st.info("""
    **Algorithm**: Random Forest Regressor
    
    **Features**:
    - Age
    - Gender  
    - Education Level
    - Job Title
    - Years of Experience
    
    **Target**: Salary ($)
    """)

    st.markdown("---")
    st.subheader("📚 How to Use")
    st.markdown("""
    1. **🏠 Home**: Overview of the dataset
    2. **🔮 Predictor**: Get salary predictions
    3. **📊 Analysis**: Explore data patterns
    4. **📈 Performance**: Model accuracy metrics
    5. **🔍 Insights**: Key findings and trends
    """)

# Add download functionality
if st.sidebar.button("📥 Download Sample Data"):
    csv = df.head(100).to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="sample_salary_data.csv",
        mime="text/csv"
    )
