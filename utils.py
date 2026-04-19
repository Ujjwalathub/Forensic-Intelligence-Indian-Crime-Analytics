"""
Utility Functions for Crime Data Analysis
Visualization, reporting, and analysis helpers
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking in Flask
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_comparison_report(juvenile_risks, victim_shifts, institutional_risks):
    """
    Create a comprehensive comparison report of all three models
    """
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RISK ANALYSIS REPORT")
    print("=" * 100)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Juvenile Recidivism Summary
    print("\n" + "-" * 100)
    print("1. JUVENILE RECIDIVISM RISK - SUMMARY")
    print("-" * 100)
    
    if juvenile_risks is not None:
        top_recidivism = juvenile_risks.groupby('Area_Name')['Risk_Score'].mean().sort_values(ascending=False).head(10)
        print("\nTop 10 States with Highest Juvenile Recidivism Risk:")
        for i, (state, risk) in enumerate(top_recidivism.items(), 1):
            print(f"  {i:2d}. {state:30s} - Risk Score: {risk:.4f}")
    
    # Victim Vulnerability Summary
    print("\n" + "-" * 100)
    print("2. VICTIM VULNERABILITY - SUMMARY")
    print("-" * 100)
    
    if victim_shifts is not None:
        increasing_shifts = victim_shifts[victim_shifts['Change_Percentage'] > 0].head(5)
        print("\nTop 5 Demographics with Increasing Victimization:")
        for i, row in increasing_shifts.iterrows():
            print(f"  • {row['Crime_Type']}: {row['Age_Group']}")
            print(f"    Change: {row['Change_Percentage']:+.1f}% "
                  f"({int(row['Early_Victims'])} → {int(row['Recent_Victims'])})")
    
    # Institutional Stress Summary
    print("\n" + "-" * 100)
    print("3. INSTITUTIONAL STRESS & EARLY WARNING - SUMMARY")
    print("-" * 100)
    
    if institutional_risks is not None:
        high_risk_states = institutional_risks[institutional_risks['Average_Risk_Score'] > 0.7]
        print(f"\nTotal States flagged as HIGH RISK (>0.7): {len(high_risk_states)}")
        print("\nTop 10 States Requiring Immediate Intervention:")
        for i, row in institutional_risks.head(10).iterrows():
            print(f"  {i+1:2d}. {row['State']:30s} - Risk Score: {row['Average_Risk_Score']:.4f}")


def plot_comparative_analysis(juvenile_risks, victim_profiles, institutional_risks):
    """
    Create comparative visualizations across all three models
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Juvenile Recidivism Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if juvenile_risks is not None:
        state_recidivism = juvenile_risks.groupby('Area_Name')['Risk_Score'].mean().sort_values(ascending=False).head(15)
        ax1.barh(state_recidivism.index, state_recidivism.values, color='steelblue')
        ax1.set_xlabel('Average Risk Score')
        ax1.set_title('Juvenile Recidivism: Top 15 States')
        ax1.invert_yaxis()
    
    # 2. Victim Counts by Crime Type
    ax2 = fig.add_subplot(gs[0, 1])
    if victim_profiles is not None:
        crime_counts = victim_profiles.groupby('Crime_Type')['Victim_Count'].sum()
        ax2.bar(crime_counts.index, crime_counts.values, color='coral')
        ax2.set_ylabel('Total Victims')
        ax2.set_title('Total Victims by Crime Type')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Institutional Risk Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if institutional_risks is not None:
        risk_dist = institutional_risks['Average_Risk_Score']
        ax3.hist(risk_dist, bins=30, color='darkred', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Risk Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Institutional Stress Risk Distribution')
        ax3.axvline(risk_dist.mean(), color='yellow', linestyle='--', linewidth=2)
    
    # 4. Top Risk States Across All Models
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['Juvenile\nRecidivism', 'Victim\nVulnerability', 'Institutional\nStress']
    if all([juvenile_risks is not None, victim_profiles is not None, institutional_risks is not None]):
        values = [
            juvenile_risks['Risk_Score'].mean(),
            victim_profiles['Percentage_of_Crime'].mean(),
            institutional_risks['Average_Risk_Score'].mean()
        ]
        ax4.bar(categories, values, color=['steelblue', 'coral', 'darkred'])
        ax4.set_ylabel('Average Risk Score')
        ax4.set_title('Model Comparison: Average Risk Scores')
        ax4.set_ylim([0, 1])
    
    # 5. Yearly Trends
    ax5 = fig.add_subplot(gs[2, :])
    if juvenile_risks is not None:
        yearly_trend = juvenile_risks.groupby('Year')['Risk_Score'].mean()
        ax5.plot(yearly_trend.index, yearly_trend.values, marker='o', linewidth=2, color='steelblue', label='Juvenile Recidivism')
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Average Risk Score')
        ax5.set_title('Risk Trends Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.savefig('comparative_risk_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparative analysis plot saved as 'comparative_risk_analysis.png'")
    plt.close()


def generate_intervention_recommendations(juvenile_risks, institutional_risks):
    """
    Generate actionable recommendations based on model outputs
    """
    print("\n" + "=" * 100)
    print("INTERVENTION RECOMMENDATIONS")
    print("=" * 100)
    
    recommendations = []
    
    # Juvenile Recidivism Recommendations
    if juvenile_risks is not None:
        high_risk_juveniles = juvenile_risks[juvenile_risks['Risk_Score'] > 0.6]
        if len(high_risk_juveniles) > 0:
            top_states = high_risk_juveniles['Area_Name'].value_counts().head(5)
            print("\n1. JUVENILE RECIDIVISM INTERVENTIONS")
            print("   Priority States for Educational Support Programs:")
            for state, count in top_states.items():
                print(f"   • {state}: {count} high-risk cases")
                recommendations.append({
                    'Type': 'Juvenile Education',
                    'Priority': 'High',
                    'State': state,
                    'Action': 'Implement vocational training and rehabilitation programs'
                })
    
    # Institutional Stress Recommendations
    if institutional_risks is not None:
        critical_states = institutional_risks[institutional_risks['Average_Risk_Score'] > 0.8]
        if len(critical_states) > 0:
            print("\n2. INSTITUTIONAL STRESS MANAGEMENT")
            print("   CRITICAL: States requiring immediate oversight:")
            for _, row in critical_states.head(5).iterrows():
                print(f"   • {row['State']}: Risk Score {row['Average_Risk_Score']:.4f}")
                recommendations.append({
                    'Type': 'Police Accountability',
                    'Priority': 'Critical',
                    'State': row['State'],
                    'Action': 'Deploy human rights monitoring, conduct independent audits'
                })
        
        medium_risk = institutional_risks[
            (institutional_risks['Average_Risk_Score'] > 0.5) & 
            (institutional_risks['Average_Risk_Score'] <= 0.8)
        ]
        if len(medium_risk) > 0:
            print("\n   HIGH: States requiring enhanced monitoring:")
            for _, row in medium_risk.head(5).iterrows():
                print(f"   • {row['State']}: Risk Score {row['Average_Risk_Score']:.4f}")
    
    return pd.DataFrame(recommendations)


def create_executive_summary(juvenile_risks, victim_shifts, institutional_risks):
    """
    Create a one-page executive summary
    """
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY - CRIME DATA PREDICTIVE MODELING")
    print("=" * 100)
    
    summary = f"""
PROJECT OVERVIEW:
This analysis used machine learning to identify three critical crime-related risks across Indian states:

1. JUVENILE RECIDIVISM RISK
   - Model: Random Forest Classifier
   - Purpose: Predict which juveniles are likely to reoffend
   - Key Finding: {juvenile_risks['Risk_Score'].mean():.1%} average recidivism risk across states
   - Recommendation: Target educational and economic interventions in high-risk states

2. VICTIM VULNERABILITY PROFILING
   - Model: Demographic Risk Scoring
   - Purpose: Identify vulnerable populations for specific crime types
   - Key Finding: {victim_shifts['Change_Percentage'].mean():.1f}% average change in victimization patterns
   - Recommendation: Deploy victim support programs in high-vulnerability regions

3. INSTITUTIONAL STRESS & EARLY WARNING SYSTEM
   - Model: Logistic Regression with ROC-AUC validation
   - Purpose: Predict districts at risk for human rights violations
   - Key Finding: {institutional_risks['Average_Risk_Score'].mean():.1%} average institutional stress level
   - Recommendation: Increase oversight and accountability measures in flagged districts

OVERALL IMPACT:
These models provide early warning capabilities for policymakers to:
✓ Allocate prevention resources efficiently
✓ Identify intervention priorities
✓ Monitor institutional health in real-time
✓ Evaluate policy effectiveness

DATA SOURCES:
- Indian Crime Statistics (2001-2014)
- 43 datasets covering: juveniles, victims, police complaints, custodial deaths, HR violations

NEXT STEPS:
1. Integrate models into dashboard for real-time monitoring
2. Validate recommendations with domain experts
3. Develop intervention protocols for high-risk areas
4. Create quarterly reporting pipeline
    """
    
    print(summary)
    
    # Save to file
    with open('executive_summary.txt', 'w', encoding='utf-8') as f:
        f.write("EXECUTIVE SUMMARY - CRIME DATA PREDICTIVE MODELING\n")
        f.write("=" * 100 + "\n")
        f.write(summary)
    
    print("\n✓ Executive summary saved to 'executive_summary.txt'")


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
