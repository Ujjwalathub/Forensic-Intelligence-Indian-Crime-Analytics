"""
Juvenile Recidivism Risk Classification Model
Predicts which juveniles are at risk of repeat offending based on background characteristics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking in Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, f1_score, precision_recall_curve, 
                             roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

from data_preparation import merge_juvenile_features


class JuvenileRecidivismModel:
    """
    Classification model to predict juvenile recidivism risk
    Features: Education, Economic Setup, Family Background
    Target: Recidivism (binary: repeat offender vs. first-time)
    """
    
    def __init__(self):
        self.model = None
        self.model_gb = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = {}
        
    def prepare_features(self, df):
        """
        Extract and engineer features from raw data
        """
        print("\n" + "=" * 80)
        print("JUVENILE RECIDIVISM: FEATURE ENGINEERING")
        print("=" * 80)
        
        df = df.copy()
        
        # Extract education features
        education_features = [col for col in df.columns if col.startswith('Education_')]
        
        # Extract economic features
        economic_features = [col for col in df.columns if col.startswith('Economic_')]
        
        # Extract family background features
        family_features = [col for col in df.columns if col.startswith('Family_')]
        
        # Extract recidivism columns
        recidivism_cols = [col for col in df.columns if col.startswith('Recidivism_')]
        
        # Create feature matrix X
        feature_cols = education_features + economic_features + family_features
        X = df[['Area_Name', 'Year'] + feature_cols].copy()
        
        # Create target y: Binary classification
        # 1 = has recidivism (Old Delinquent), 0 = first-time offender (New Delinquent)
        if 'Recidivism_Old_Delinquent' in df.columns:
            y = (df['Recidivism_Old_Delinquent'] > 0).astype(int)
        else:
            y = pd.Series(0, index=df.index)
        
        # Remove rows with all zero features
        X_features = X[feature_cols]
        valid_rows = (X_features.sum(axis=1) > 0)
        
        X = X[valid_rows].reset_index(drop=True)
        y = y[valid_rows].reset_index(drop=True)
        
        # Extract metadata
        metadata = X[['Area_Name', 'Year']].copy()
        X_features = X[feature_cols].copy()
        
        print(f"✓ Feature matrix shape: {X_features.shape}")
        print(f"✓ Target distribution: {y.value_counts().to_dict()}")
        print(f"✓ Recidivism rate: {y.mean():.2%}")
        print(f"✓ Features used: {len(feature_cols)}")
        
        return X_features, y, metadata, feature_cols
    
    def train_model(self, X, y, metadata):
        """
        Train Random Forest and Gradient Boosting models
        """
        print("\n" + "=" * 80)
        print("JUVENILE RECIDIVISM: MODEL TRAINING")
        print("=" * 80)
        
        # Split data
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, metadata, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Train set size: {X_train.shape[0]}")
        print(f"📊 Test set size: {X_test.shape[0]}")
        print(f"✓ Train recidivism rate: {y_train.mean():.2%}")
        print(f"✓ Test recidivism rate: {y_test.mean():.2%}")
        
        # Train Random Forest
        print("\n🔧 Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Train Gradient Boosting
        print("🔧 Training Gradient Boosting Classifier...")
        self.model_gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model_gb.fit(X_train, y_train)
        
        # Predictions
        y_pred_rf = self.model.predict(X_test)
        y_pred_proba_rf = self.model.predict_proba(X_test)[:, 1]
        
        y_pred_gb = self.model_gb.predict(X_test)
        y_pred_proba_gb = self.model_gb.predict_proba(X_test)[:, 1]
        
        # Evaluate
        self.metrics['random_forest'] = self._evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')
        self.metrics['gradient_boosting'] = self._evaluate_model(y_test, y_pred_gb, y_pred_proba_gb, 'Gradient Boosting')
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'RF_Importance': self.model.feature_importances_,
            'GB_Importance': self.model_gb.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        return meta_test, y_test, y_pred_rf, y_pred_proba_rf
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Evaluate model performance
        """
        print(f"\n{model_name} Results:")
        print("-" * 40)
        
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['No Recidivism', 'Recidivism']))
        
        return metrics
    
    def plot_feature_importance(self, top_n=15):
        """
        Visualize feature importance
        """
        if self.feature_importance is None:
            print("Model not trained yet!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Random Forest
        top_features_rf = self.feature_importance.nlargest(top_n, 'RF_Importance')
        axes[0].barh(top_features_rf['Feature'], top_features_rf['RF_Importance'], color='steelblue')
        axes[0].set_xlabel('Importance Score')
        axes[0].set_title('Random Forest - Top 15 Features')
        axes[0].invert_yaxis()
        
        # Gradient Boosting
        top_features_gb = self.feature_importance.nlargest(top_n, 'GB_Importance')
        axes[1].barh(top_features_gb['Feature'], top_features_gb['GB_Importance'], color='darkorange')
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('Gradient Boosting - Top 15 Features')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('juvenile_recidivism_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved")
        plt.close()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Recidivism', 'Recidivism'],
                   yticklabels=['No Recidivism', 'Recidivism'])
        plt.title('Juvenile Recidivism - Confusion Matrix (Random Forest)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('juvenile_recidivism_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix plot saved")
        plt.close()


def run_juvenile_recidivism_analysis():
    """
    Execute complete juvenile recidivism analysis
    """
    print("\n" + "=" * 80)
    print("JUVENILE RECIDIVISM RISK CLASSIFICATION")
    print("=" * 80)
    
    # Load and prepare data
    df = merge_juvenile_features()
    
    # Initialize model
    model = JuvenileRecidivismModel()
    
    # Prepare features
    X, y, metadata, feature_cols = model.prepare_features(df)
    
    # Train model
    meta_test, y_test, y_pred, y_pred_proba = model.train_model(X, y, metadata)
    
    # Visualizations
    model.plot_feature_importance()
    model.plot_confusion_matrix(y_test, y_pred)
    
    # Create risk scores for states
    print("\n" + "=" * 80)
    print("STATE-LEVEL RECIDIVISM RISK SCORES")
    print("=" * 80)
    
    state_risk_scores = meta_test.copy()
    state_risk_scores['Risk_Score'] = y_pred_proba
    state_risk_scores['Prediction'] = y_pred
    
    state_summary = state_risk_scores.groupby('Area_Name').agg({
        'Risk_Score': ['mean', 'max', 'min', 'std'],
        'Prediction': 'mean'
    }).round(4)
    
    state_summary.columns = ['Avg_Risk', 'Max_Risk', 'Min_Risk', 'Risk_Std', 'Recidivism_Rate']
    state_summary = state_summary.sort_values('Avg_Risk', ascending=False)
    
    print("\nTop 10 Highest Risk States:")
    print(state_summary.head(10))
    
    # Save results
    state_summary.to_csv('juvenile_recidivism_state_risk.csv')
    print("\n✓ State-level risk scores saved to 'juvenile_recidivism_state_risk.csv'")
    
    return model, state_risk_scores


if __name__ == "__main__":
    model, risk_scores = run_juvenile_recidivism_analysis()
