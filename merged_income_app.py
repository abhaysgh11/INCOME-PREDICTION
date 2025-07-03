#  streamlit run .\frontend.py 

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'income']
    df.columns = col_names

    # Remove whitespace
    df = df.applymap(lambda x: x.strip() if type(x)==str else x)

    # Encode categorical variables
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])

    X = df.drop('income', axis=1)
    y = df['income']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Load and train model once
X_train, X_test, y_train, y_test = load_and_preprocess_data('income_evaluation.csv')
model = train_model(X_train, y_train)

# Predict function for Streamlit use
def predict_income(input_df):
    return model.predict(input_df)[0]

# Configure page
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        background-color: #f0f8ff;
        text-align: center;
        margin: 1rem 0;
    }
    .high-income {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    .low-income {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the dataset for training"""
    # For demo purposes, we'll create sample data similar to the census income dataset
    # In practice, you would load your actual CSV file here
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'age': np.random.randint(17, 90, n_samples),
        'education_num': np.random.randint(1, 16, n_samples),
        'capital_gain': np.random.exponential(500, n_samples),
        'capital_loss': np.random.exponential(100, n_samples),
        'hours_per_week': np.random.normal(40, 10, n_samples).clip(1, 99),
        'workclass_ Private': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    }
    
    # Create target variable with some logic
    income_prob = (
        (data['age'] - 25) * 0.01 +
        data['education_num'] * 0.05 +
        data['capital_gain'] * 0.0001 +
        data['hours_per_week'] * 0.01 +
        data['workclass_ Private'] * 0.2 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    data['income_ >50K'] = (income_prob > 0.5).astype(int)
    
    return pd.DataFrame(data)

@st.cache_resource
def train_models(df):
    """Train all models and return them"""
    # Prepare features and target
    X = df.drop(columns=["income_ >50K"])
    y = df["income_ >50K"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 2},
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # XGBoost
    xgb_model = XGBClassifier(
        scale_pos_weight=3,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True
    )
    stacking_model.fit(X_train, y_train)
    models['Stacking Ensemble'] = stacking_model
    
    # Calculate accuracies
    accuracies = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)
    
    return models, accuracies, X_test, y_test

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Feature Importance',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        return fig
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Income Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load and prepare data
    with st.spinner("Loading data and training models..."):
        df = load_and_prepare_data()
        models, accuracies, X_test, y_test = train_models(df)
    
    # Sidebar for user input
    st.sidebar.markdown('<h2 class="sub-header">📊 Enter Your Information</h2>', unsafe_allow_html=True)
    
    # User inputs
    age = st.sidebar.slider("Age", min_value=17, max_value=90, value=35, step=1)
    education_num = st.sidebar.slider("Education Level (Years)", min_value=1, max_value=16, value=12, step=1)
    capital_gain = st.sidebar.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.sidebar.number_input("Capital Loss ($)", min_value=0, max_value=10000, value=0, step=50)
    hours_per_week = st.sidebar.slider("Hours per Week", min_value=1, max_value=99, value=40, step=1)
    workclass_private = st.sidebar.selectbox("Work in Private Sector?", ["No", "Yes"])
    
    # Convert inputs to model format
    user_data = pd.DataFrame({
        'age': [age],
        'education_num': [education_num],
        'capital_gain': [capital_gain],
        'capital_loss': [capital_loss],
        'hours_per_week': [hours_per_week],
        'workclass_ Private': [1 if workclass_private == "Yes" else 0]
    })
    
    # Model selection
    st.sidebar.markdown("---")
    selected_model = st.sidebar.selectbox(
        "Choose Prediction Model",
        list(models.keys()),
        index=3  # Default to Stacking Ensemble
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🔮 Income Prediction</h2>', unsafe_allow_html=True)
        
        # Make prediction
        model = models[selected_model]
        prediction = model.predict(user_data)[0]
        prediction_proba = model.predict_proba(user_data)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        # Display prediction
        if prediction == 1:
            st.markdown(f'''
            <div class="prediction-box high-income">
                <h3>🎉 Predicted Income: >$50K</h3>
                <p>Confidence: {prediction_proba[1]:.2%}</p>
                <p>Model: {selected_model}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-box low-income">
                <h3>📊 Predicted Income: ≤$50K</h3>
                <p>Confidence: {prediction_proba[0]:.2%}</p>
                <p>Model: {selected_model}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Probability visualization
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=['≤$50K', '>$50K'],
            y=[prediction_proba[0], prediction_proba[1]],
            marker_color=['#ff7f7f', '#7fbf7f'],
            text=[f'{prediction_proba[0]:.2%}', f'{prediction_proba[1]:.2%}'],
            textposition='auto'
        ))
        fig_prob.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Model Performance</h2>', unsafe_allow_html=True)
        
        # Model accuracies
        acc_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
        fig_acc = px.bar(
            acc_df, 
            x='Accuracy', 
            y='Model',
            orientation='h',
            color='Accuracy',
            color_continuous_scale='blues',
            text='Accuracy'
        )
        fig_acc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_acc.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Current model accuracy
        current_accuracy = accuracies[selected_model]
        st.metric(
            label=f"{selected_model} Accuracy",
            value=f"{current_accuracy:.3f}",
            delta=f"{current_accuracy - min(accuracies.values()):.3f}"
        )
    
    # Feature Analysis Section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📈 Feature Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Your Profile", "🔍 Feature Importance", "📋 Data Overview"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Age", f"{age} years")
            st.metric("Education", f"{education_num} years")
        
        with col2:
            st.metric("Capital Gain", f"${capital_gain:,}")
            st.metric("Capital Loss", f"${capital_loss:,}")
        
        with col3:
            st.metric("Hours/Week", f"{hours_per_week} hrs")
            st.metric("Private Sector", workclass_private)
        
        # User profile radar chart
        categories = ['Age (scaled)', 'Education', 'Capital Gain (scaled)', 
                     'Hours/Week (scaled)', 'Private Sector']
        values = [
            age/90,  # Scale to 0-1
            education_num/16,
            min(capital_gain/10000, 1),  # Cap at 1
            hours_per_week/99,
            1 if workclass_private == "Yes" else 0
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Profile'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Your Profile Visualization"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        # Feature importance for tree-based models
        if selected_model in ['Random Forest', 'XGBoost']:
            fig_importance = create_feature_importance_plot(
                models[selected_model], 
                user_data.columns
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance is only available for tree-based models (Random Forest, XGBoost)")
    
    with tab3:
        st.write("**Dataset Overview:**")
        st.write(f"- Total samples: {len(df):,}")
        st.write(f"- High income rate: {df['income_ >50K'].mean():.2%}")
        
        # Distribution plots
        fig_dist = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Age', 'Education', 'Capital Gain', 
                          'Hours/Week', 'Capital Loss', 'Work Class'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Age distribution
        fig_dist.add_trace(
            go.Histogram(x=df['age'], name='Age', nbinsx=20),
            row=1, col=1
        )
        
        # Education distribution
        fig_dist.add_trace(
            go.Histogram(x=df['education_num'], name='Education', nbinsx=16),
            row=1, col=2
        )
        
        # Capital gain distribution (log scale for better visualization)
        fig_dist.add_trace(
            go.Histogram(x=np.log1p(df['capital_gain']), name='Log Capital Gain', nbinsx=20),
            row=1, col=3
        )
        
        # Hours per week distribution
        fig_dist.add_trace(
            go.Histogram(x=df['hours_per_week'], name='Hours/Week', nbinsx=20),
            row=2, col=1
        )
        
        # Capital loss distribution
        fig_dist.add_trace(
            go.Histogram(x=np.log1p(df['capital_loss']), name='Log Capital Loss', nbinsx=20),
            row=2, col=2
        )
        
        # Work class distribution
        work_class_counts = df['workclass_ Private'].value_counts()
        fig_dist.add_trace(
            go.Bar(x=['Government/Other', 'Private'], y=[work_class_counts[0], work_class_counts[1]], name='Work Class'),
            row=2, col=3
        )
        
        fig_dist.update_layout(height=600, showlegend=False, title_text="Data Distribution Overview")
        st.plotly_chart(fig_dist, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit | Machine Learning Income Prediction</p>
        <p><em>This app uses ensemble machine learning models to predict income levels based on demographic and work-related features.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()