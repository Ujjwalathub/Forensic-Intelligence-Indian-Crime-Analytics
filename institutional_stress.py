"""
Institutional Stress & Early Warning System (EWS)
Predicts risk of human rights violations and custodial deaths based on complaint patterns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking in Flask
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, f1_score, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

from data_preparation import merge_institutional_stress_features


class InstitutionalStressEWS:
    """
    Early Warning System for institutional stress and human rights violations
    Uses complaints as leading indicators for custodial deaths and HR violations
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.metrics = {}
        
    def prepare_features(self, df):
        """
        Extract features from complaints and HR violation data
        """
        print("\n" + "=" * 80)
        print("INSTITUTIONAL STRESS: FEATURE PREPARATION")
        print("=" * 80)
        
        df = df.copy()
        df = df.fillna(0)
        
        # Extract complaint-related columns (leading indicators)
        complaint_cols = [col for col in df.columns if col.startswith('CPA_')]
        
        # Extract HR violation columns
        hr_cols = [col for col in df.columns if col not in ['Area_Name', 'Year'] 
                  and col.startswith('HR_') or any(x in col for x in ['Illegal', 'Torture', 'Encounter'])]
        
        # If specific HR columns don't exist, create aggregate metrics
        if len(hr_cols) == 0:
            # Create stress indicators from available columns
            feature_cols = complaint_cols[:10] if complaint_cols else []
        else:
            feature_cols = complaint_cols + hr_cols
        
        print(f"✓ Feature columns identified: {len(feature_cols)}")
        
        # Handle missing features gracefully
        available_cols = [col for col in feature_cols if col in df.columns]
        feature_cols = available_cols[:20]  # Limit to 20 features for interpretability
        
        X = df[['Area_Name', 'Year'] + feature_cols].copy()
        
        # Create binary target: High Risk if any complaints or violations exist
        # Target: 1 if complaints + violations exceed threshold, 0 otherwise
        if complaint_cols:
            complaint_sum = df[[col for col in complaint_cols if col in df.columns]].sum(axis=1)
        else:
            complaint_sum = pd.Series(0, index=df.index)
        
        # Use quantile-based threshold
        threshold = complaint_sum.quantile(0.75)
        y = (complaint_sum > threshold).astype(int)
        
        # Remove rows with all zero features
        X_features = X[feature_cols]
        valid_rows = (X_features.sum(axis=1) >= 0)  # Keep all rows for EWS
        
        X = X[valid_rows].reset_index(drop=True)
        y = y[valid_rows].reset_index(drop=True)
        
        metadata = X[['Area_Name', 'Year']].copy()
        X_features = X[feature_cols].copy()
        
        # Normalize features
        X_features = (X_features - X_features.mean()) / (X_features.std() + 1e-8)
        X_features = X_features.fillna(0)
        
        print(f"✓ Feature matrix shape: {X_features.shape}")
        print(f"✓ Target distribution - High Risk: {y.sum()} ({y.mean():.2%})")
        print(f"✓ Features used: {len(feature_cols)}")
        
        return X_features, y, metadata, feature_cols
    
    def train_model(self, X, y, metadata):
        """
        Train logistic regression and ensemble models for EWS
        """
        print("\n" + "=" * 80)
        print("INSTITUTIONAL STRESS: MODEL TRAINING")
        print("=" * 80)
        
        # Split data with temporal consideration
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, metadata, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Train set size: {X_train.shape[0]}")
        print(f"📊 Test set size: {X_test.shape[0]}")
        print(f"✓ Train high-risk rate: {y_train.mean():.2%}")
        print(f"✓ Test high-risk rate: {y_test.mean():.2%}")
        
        # Train Logistic Regression
        print("\n🔧 Training Logistic Regression (EWS)...")
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred_lr = self.model.predict(X_test)
        y_pred_proba_lr = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics['logistic_regression'] = self._evaluate_model(
            y_test, y_pred_lr, y_pred_proba_lr, 'Logistic Regression'
        )
        
        # Train Random Forest for comparison
        print("\n🔧 Training Random Forest (for comparison)...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        y_pred_rf = rf_model.predict(X_test)
        y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        
        self.metrics['random_forest'] = self._evaluate_model(
            y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest'
        )
        
        # Store feature importance from LR (coefficients)
        self.feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': self.model.coef_[0],
            'Abs_Coefficient': np.abs(self.model.coef_[0]),
            'RF_Importance': rf_model.feature_importances_
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return meta_test, y_test, y_pred_proba_lr, y_pred_proba_rf
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Evaluate EWS model performance
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
                                   target_names=['Low Risk', 'High Risk']))
        
        return metrics
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot most important features for EWS
        """
        if self.feature_importance is None:
            print("Model not trained yet!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Logistic Regression Coefficients
        top_features_lr = self.feature_importance.nlargest(top_n, 'Abs_Coefficient')
        colors_lr = ['green' if x > 0 else 'red' for x in top_features_lr['Coefficient']]
        axes[0].barh(top_features_lr['Feature'], top_features_lr['Coefficient'], color=colors_lr)
        axes[0].set_xlabel('Coefficient Value')
        axes[0].set_title('Logistic Regression - Impact on High Risk Classification')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0].invert_yaxis()
        
        # Random Forest Feature Importance
        top_features_rf = self.feature_importance.nlargest(top_n, 'RF_Importance')
        axes[1].barh(top_features_rf['Feature'], top_features_rf['RF_Importance'], color='steelblue')
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('Random Forest - Feature Importance for EWS')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('institutional_stress_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved")
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_lr, y_pred_rf):
        """
        Plot ROC curve for EWS model
        """
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
        roc_auc_lr = auc(fpr_lr, tpr_lr)
        
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
        plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Early Warning System - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('institutional_stress_roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve plot saved")
        plt.close()
    
    def identify_high_risk_districts(self, metadata, y_pred_proba):
        """
        Identify high-risk districts for intervention
        """
        risk_scores = metadata.copy()
        risk_scores['Risk_Probability'] = y_pred_proba
        
        # Aggregate by state and year
        state_risk = risk_scores.groupby(['Area_Name', 'Year']).agg({
            'Risk_Probability': 'mean'
        }).reset_index()
        
        state_risk.columns = ['State', 'Year', 'Average_Risk_Score']
        state_risk = state_risk.sort_values('Average_Risk_Score', ascending=False)
        
        return state_risk
    
    def plot_risk_distribution(self, state_risk):
        """
        Visualize risk distribution across states
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 20 highest risk states
        top_states = state_risk.groupby('State')['Average_Risk_Score'].mean().sort_values(ascending=False).head(20)
        axes[0].barh(top_states.index, top_states.values, color='darkred')
        axes[0].set_xlabel('Average Risk Score')
        axes[0].set_title('Top 20 Highest Risk States (EWS)')
        axes[0].invert_yaxis()
        
        # Risk score distribution histogram
        axes[1].hist(state_risk['Average_Risk_Score'], bins=30, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('Risk Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Institutional Stress Risk Scores')
        axes[1].axvline(state_risk['Average_Risk_Score'].mean(), color='red', 
                       linestyle='--', linewidth=2, label='Mean')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('institutional_stress_risk_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Risk distribution plot saved")
        plt.close()


def run_institutional_stress_analysis():
    """
    Execute complete institutional stress EWS analysis
    """
    print("\n" + "=" * 80)
    print("INSTITUTIONAL STRESS & EARLY WARNING SYSTEM (EWS)")
    print("=" * 80)
    
    # Load and prepare data
    df = merge_institutional_stress_features()
    
    # Initialize model
    ews = InstitutionalStressEWS()
    
    # Prepare features
    X, y, metadata, feature_cols = ews.prepare_features(df)
    
    # Train model
    meta_test, y_test, y_pred_proba_lr, y_pred_proba_rf = ews.train_model(X, y, metadata)
    
    # Identify high-risk districts
    state_risk = ews.identify_high_risk_districts(meta_test, y_pred_proba_lr)
    
    print("\n" + "=" * 80)
    print("HIGH-RISK STATES FOR INTERVENTION")
    print("=" * 80)
    print("\nTop 15 States with Highest Institutional Stress:")
    print(state_risk.head(15).to_string(index=False))
    
    # Visualizations
    ews.plot_feature_importance()
    ews.plot_roc_curve(y_test, y_pred_proba_lr, y_pred_proba_rf)
    ews.plot_risk_distribution(state_risk)
    
    # Save results
    state_risk.to_csv('institutional_stress_state_risk.csv', index=False)
    ews.feature_importance.to_csv('ews_feature_importance.csv', index=False)
    
    print("\n✓ Results saved:")
    print("  - institutional_stress_state_risk.csv")
    print("  - ews_feature_importance.csv")
    
    return ews, state_risk


if __name__ == "__main__":
    ews, state_risk = run_institutional_stress_analysis()
