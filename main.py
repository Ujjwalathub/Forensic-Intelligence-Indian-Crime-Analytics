"""
Main Orchestration Script
Runs all three predictive models and generates comprehensive analysis reports
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from juvenile_recidivism import run_juvenile_recidivism_analysis
from victim_vulnerability import run_victim_vulnerability_analysis
from institutional_stress import run_institutional_stress_analysis
from utils import (create_comparison_report, plot_comparative_analysis, 
                   generate_intervention_recommendations, create_executive_summary)


def main():
    """
    Execute all three predictive models in sequence
    """
    print("\n" + "=" * 100)
    print("INDIAN CRIME DATASET: COMPREHENSIVE PREDICTIVE MODELING ANALYSIS")
    print("=" * 100)
    print("\nThis analysis implements three machine learning models to predict:")
    print("  1. Juvenile Recidivism Risk - Classification")
    print("  2. Victim Vulnerability Profiling - Risk Scoring")
    print("  3. Institutional Stress & Early Warning System - Classification")
    print("\n" + "=" * 100)
    
    try:
        # ============================================================================
        # MODEL 1: JUVENILE RECIDIVISM RISK CLASSIFICATION
        # ============================================================================
        print("\n\n🚀 STARTING MODEL 1: JUVENILE RECIDIVISM RISK CLASSIFICATION...")
        print("-" * 100)
        
        juvenile_model, juvenile_risks = run_juvenile_recidivism_analysis()
        
        print("\n✅ Juvenile Recidivism Model Completed")
        print(f"   ✓ Total risk assessments: {len(juvenile_risks)}")
        print(f"   ✓ Average risk score: {juvenile_risks['Risk_Score'].mean():.4f}")
        print(f"   ✓ States analyzed: {juvenile_risks['Area_Name'].nunique()}")
        
        # ============================================================================
        # MODEL 2: VICTIM VULNERABILITY PROFILING
        # ============================================================================
        print("\n\n🚀 STARTING MODEL 2: VICTIM VULNERABILITY PROFILING...")
        print("-" * 100)
        
        vulnerability_analysis, victim_profiles, victim_shifts = run_victim_vulnerability_analysis()
        
        print("\n✅ Victim Vulnerability Model Completed")
        print(f"   ✓ Total victim profiles: {len(victim_profiles)}")
        print(f"   ✓ Demographic shifts identified: {len(victim_shifts)}")
        print(f"   ✓ Crime types analyzed: {victim_profiles['Crime_Type'].nunique()}")
        
        # ============================================================================
        # MODEL 3: INSTITUTIONAL STRESS & EARLY WARNING SYSTEM
        # ============================================================================
        print("\n\n🚀 STARTING MODEL 3: INSTITUTIONAL STRESS & EARLY WARNING SYSTEM...")
        print("-" * 100)
        
        ews_model, institutional_risks = run_institutional_stress_analysis()
        
        print("\n✅ Institutional Stress EWS Model Completed")
        print(f"   ✓ Risk assessments: {len(institutional_risks)}")
        print(f"   ✓ Average risk score: {institutional_risks['Average_Risk_Score'].mean():.4f}")
        print(f"   ✓ Critical risk states: {len(institutional_risks[institutional_risks['Average_Risk_Score'] > 0.8])}")
        
        # ============================================================================
        # COMPREHENSIVE ANALYSIS & REPORTING
        # ============================================================================
        print("\n\n📊 GENERATING COMPREHENSIVE ANALYSIS & REPORTS...")
        print("-" * 100)
        
        # Create comparison report
        create_comparison_report(juvenile_risks, victim_shifts, institutional_risks)
        
        # Generate visualizations
        print("\n🎨 Creating comparative visualizations...")
        plot_comparative_analysis(juvenile_risks, victim_profiles, institutional_risks)
        
        # Generate intervention recommendations
        print("\n💡 Generating intervention recommendations...")
        recommendations = generate_intervention_recommendations(juvenile_risks, institutional_risks)
        recommendations.to_csv('intervention_recommendations.csv', index=False)
        print("✓ Intervention recommendations saved")
        
        # Create executive summary
        print("\n📋 Creating executive summary...")
        create_executive_summary(juvenile_risks, victim_shifts, institutional_risks)
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        print("\n\n" + "=" * 100)
        print("ANALYSIS COMPLETE - SUMMARY OF OUTPUTS")
        print("=" * 100)
        
        print("\n📁 OUTPUT FILES GENERATED:")
        print("\n   Model Results:")
        print("   ✓ juvenile_recidivism_state_risk.csv")
        print("   ✓ victim_vulnerability_profiles.csv")
        print("   ✓ victim_state_vulnerability.csv")
        print("   ✓ victim_demographic_shifts.csv")
        print("   ✓ institutional_stress_state_risk.csv")
        print("   ✓ ews_feature_importance.csv")
        
        print("\n   Feature Analysis:")
        print("   ✓ juvenile_recidivism_feature_importance.png")
        print("   ✓ juvenile_recidivism_confusion_matrix.png")
        print("   ✓ victim_vulnerability_heatmap.png")
        print("   ✓ victim_temporal_trends.png")
        print("   ✓ institutional_stress_feature_importance.png")
        print("   ✓ institutional_stress_roc_curve.png")
        print("   ✓ institutional_stress_risk_distribution.png")
        print("   ✓ comparative_risk_analysis.png")
        
        print("\n   Reports & Recommendations:")
        print("   ✓ intervention_recommendations.csv")
        print("   ✓ executive_summary.txt")
        
        print("\n" + "=" * 100)
        print("🎉 ALL MODELS SUCCESSFULLY COMPLETED!")
        print("=" * 100)
        
        print("\n📈 KEY INSIGHTS:")
        print(f"\n   • Juvenile Recidivism: {(juvenile_risks['Prediction'].sum() / len(juvenile_risks) * 100):.1f}% of regions flagged")
        print(f"   • Victim Vulnerability: {len(victim_shifts[victim_shifts['Change_Percentage'] > 0])} demographics show increasing victimization")
        print(f"   • Institutional Stress: {len(institutional_risks[institutional_risks['Average_Risk_Score'] > 0.7])} states flagged as high-risk")
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Review intervention_recommendations.csv for priority actions")
        print("   2. Examine state-specific risk scores in CSV outputs")
        print("   3. Use visualizations in presentations to stakeholders")
        print("   4. Deploy models for real-time monitoring")
        
        return True
        
    except Exception as e:
        print("\n❌ ERROR OCCURRED DURING ANALYSIS:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
