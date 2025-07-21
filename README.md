# Employee Salary Prediction System

A comprehensive machine learning project that predicts employee salaries based on various factors like experience, education, job title, and demographics. Built with Python, scikit-learn, and Streamlit.

## Features

- **Machine Learning Model**: Random Forest Regressor with preprocessing pipeline
- **Interactive Web App**: Streamlit dashboard for predictions and data analysis
- **Data Visualization**: Interactive charts and graphs using Plotly
- **Model Performance**: Detailed metrics and evaluation
- **Real-time Predictions**: Input your details and get instant salary predictions

## 📊 Dataset Features

The model uses the following features to predict salaries:

- **Age**: Employee age (18-70 years)
- **Gender**: Male/Female
- **Education Level**: High School, Bachelor's, Master's, PhD
- **Job Title**: Engineer, Manager, Analyst, Developer
- **Years of Experience**: 0-40 years

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd salary-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Data
Place your `Salary Data.csv` file in the project directory, or the app will generate sample data automatically.

## Usage

### Train the Model
Run the enhanced training script to create and save your model:

```bash
python train_model.py
```

This will:
- Load and analyze your data
- Train a Random Forest model
- Display detailed performance metrics
- Save the trained model as `salary_predictor_model_enhanced.joblib`

### Launch the Web App
Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## App Interface

### Sidebar - Salary Predictor
- Input your personal details
- Get instant salary predictions
- Clean, intuitive interface

### Main Dashboard - 3 Tabs

#### 📊 Data Overview
- Dataset statistics and metrics
- Sample data preview
- Salary distribution histogram

#### 📈 Analysis
- Salary analysis by education level
- Experience vs salary correlation
- Job title compensation comparison

#### Model Performance
- R² Score and RMSE metrics
- Predicted vs Actual scatter plot
- Model accuracy visualization

## Model Details

### Algorithm
- **Random Forest Regressor**
- 100 estimators
- Max depth: 10
- Cross-validation ready

### Preprocessing Pipeline
- **Numerical Features**: Median imputation
- **Categorical Features**: Most frequent imputation + One-hot encoding
- **Scaling**: Handled automatically by Random Forest

### Performance Metrics
- **R² Score**: Model explanation power
- **RMSE**: Root Mean Square Error in dollars
- **MAE**: Mean Absolute Error

## 📁 Project Structure

```
salary-prediction/
│
├── train_model.py              # Enhanced model training script
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Salary Data.csv            # Your dataset (optional)
├── salary_predictor_model_enhanced.joblib  # Trained model
├── feature_importance.csv      # Feature analysis results
└── model_metrics.csv          # Performance metrics
```

## Requirements

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.0
plotly==5.15.0
seaborn==0.12.2
matplotlib==3.7.2
```

## 📈 Model Performance

Typical performance metrics on salary prediction:

- **R² Score**: ~0.85-0.90 (explains 85-90% of variance)
- **RMSE**: ~$8,000-12,000 (average prediction error)
- **Accuracy**: ~85-90% (within reasonable salary ranges)

## Usage Examples

### Making Predictions
```python
# Example input
input_data = {
    'Age': 35,
    'Gender': 'Male',
    'Education Level': 'Master\'s',
    'Job Title': 'Engineer',
    'Years of Experience': 10
}

# Expected prediction: ~$75,000-85,000
```

### Key Insights
- **Education Impact**: PhD adds ~$25k, Master's ~$15k premium
- **Experience Premium**: ~$2,500 per year of experience
- **Job Roles**: Engineers and Managers typically earn more
- **Age Factor**: Salary tends to increase with age and experience

## Business Use Cases

### For HR Departments
- **Salary Benchmarking**: Compare offers with market standards
- **Budget Planning**: Estimate compensation costs for new hires
- **Pay Equity Analysis**: Ensure fair compensation across demographics

### For Employees
- **Salary Negotiation**: Know your market value
- **Career Planning**: Understand salary growth potential
- **Skill Development**: Identify high-value career paths

### For Recruiters
- **Competitive Offers**: Set attractive salary packages
- **Market Research**: Stay updated on compensation trends
- **Candidate Expectations**: Align offers with experience levels

## Future Enhancements

- [ ] Add more job categories and industries
- [ ] Include location-based salary adjustments
- [ ] Implement deep learning models (Neural Networks)
- [ ] Add salary trend predictions over time
- [ ] Include skill-based salary components
- [ ] Export prediction reports to PDF

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Happy Predicting!**

*Built with ❤️ using Python, scikit-learn, and Streamlit*
