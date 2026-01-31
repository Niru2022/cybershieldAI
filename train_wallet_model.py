import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

def generate_training_data():
    """
    Generate synthetic training data representing wallet behaviors
    with better risk patterns aligned with unified risk calculation
    """
    np.random.seed(42)
    n_samples = 5000  # Increased samples for better balance
    
    data = {
        'transaction_count': np.random.poisson(50, n_samples),
        'unique_counterparties': np.random.poisson(15, n_samples),
        'avg_transaction_value': np.random.exponential(1000, n_samples),
        'balance_eth': np.random.exponential(50, n_samples),
        'transaction_velocity': np.random.normal(10, 5, n_samples),
        'amount_std_dev': np.random.exponential(500, n_samples),
        'time_between_tx_hours': np.random.exponential(6, n_samples),
        'high_value_tx_ratio': np.random.beta(2, 10, n_samples),
        'counterparty_entropy': np.random.exponential(1, n_samples),
        'is_risky': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create risk patterns aligned with unified risk calculation
    # CRITICAL risk patterns (score >= 8)
    critical_conditions = (
        (df['transaction_count'] > 150) & 
        (df['unique_counterparties'] < 3) & 
        (df['high_value_tx_ratio'] > 0.4)
    ) | (
        (df['amount_std_dev'] > 3000) & 
        (df['transaction_velocity'] > 25) & 
        (df['counterparty_entropy'] < 0.3)
    )
    
    # HIGH risk patterns (score >= 6)
    high_conditions = (
        (df['transaction_count'] > 100) & 
        (df['unique_counterparties'] < 5) & 
        (df['high_value_tx_ratio'] > 0.3)
    ) | (
        (df['amount_std_dev'] > 2000) & 
        (df['transaction_velocity'] > 20) & 
        (df['counterparty_entropy'] < 0.5)
    ) | (
        (df['balance_eth'] > 200) & 
        (df['transaction_count'] < 10)
    )
    
    # MEDIUM risk patterns (score >= 4)
    medium_conditions = (
        (df['transaction_count'] > 80) & 
        (df['unique_counterparties'] < 8) & 
        (df['high_value_tx_ratio'] > 0.2)
    ) | (
        (df['amount_std_dev'] > 1500) & 
        (df['transaction_velocity'] > 15) & 
        (df['counterparty_entropy'] < 0.7)
    )
    
    # Assign risk scores based on conditions
    df['risk_score'] = 0
    df.loc[medium_conditions, 'risk_score'] = 4
    df.loc[high_conditions, 'risk_score'] = 6
    df.loc[critical_conditions, 'risk_score'] = 8
    
    # Add some random variation to risk scores
    df['risk_score'] += np.random.randint(0, 3, n_samples)
    df['risk_score'] = np.minimum(df['risk_score'], 10)
    
    # Convert to binary classification (risky if score >= 4)
    df['is_risky'] = (df['risk_score'] >= 4).astype(int)
    
    # Ensure better class balance
    risky_count = df['is_risky'].sum()
    target_risky = int(n_samples * 0.35)  # Aim for 35% risky samples
    
    if risky_count < target_risky:
        # Add more risky samples
        safe_indices = df[df['is_risky'] == 0].index
        additional_risky = np.random.choice(
            safe_indices, 
            size=target_risky - risky_count, 
            replace=False
        )
        df.loc[additional_risky, 'is_risky'] = 1
        df.loc[additional_risky, 'risk_score'] = np.random.randint(4, 11, size=len(additional_risky))
    
    print(f"📊 Class distribution: {df['is_risky'].value_counts().to_dict()}")
    print(f"📈 Risk score distribution:")
    print(df['risk_score'].value_counts().sort_index())
    
    return df

def train_xgboost_model():
    """Train XGBoost model for wallet risk prediction"""
    
    # Generate training data
    print("🔄 Generating training data...")
    df = generate_training_data()
    
    # Features and target
    feature_columns = [
        'transaction_count', 'unique_counterparties', 'avg_transaction_value',
        'balance_eth', 'transaction_velocity', 'amount_std_dev',
        'time_between_tx_hours', 'high_value_tx_ratio', 'counterparty_entropy'
    ]
    
    X = df[feature_columns]
    y = df['is_risky']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model
    print("🏋️ Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Handle class imbalance
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model and feature information
    joblib.dump(model, 'wallet_risk_xgboost_model.pkl')
    
    # Convert numpy types to Python native types for JSON serialization
    feature_info = {
        'feature_columns': feature_columns,
        'feature_importance': dict(zip(feature_columns, [float(x) for x in model.feature_importances_])),
        'training_accuracy': float(accuracy),
        'class_distribution': {
            'non_risky': int(len(y_train[y_train == 0])),
            'risky': int(len(y_train[y_train == 1]))
        }
    }
    
    with open('model_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("💾 Model saved as 'wallet_risk_xgboost_model.pkl'")
    print("📋 Feature information saved as 'model_features.json'")
    
    # Show feature importance
    print("\n🎯 Feature Importance:")
    for feature, importance in sorted(feature_info['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    return model, feature_info

if __name__ == "__main__":
    train_xgboost_model()