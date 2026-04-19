"""
Test Suite
Validates all modules can run without errors
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "=" * 80)
    print("TEST 1: Module Imports")
    print("=" * 80)
    
    try:
        import data_preparation
        print("✓ data_preparation imported")
    except Exception as e:
        print(f"✗ data_preparation: {e}")
        return False
    
    try:
        import config
        print("✓ config imported")
    except Exception as e:
        print(f"✗ config: {e}")
        return False
    
    try:
        import juvenile_recidivism
        print("✓ juvenile_recidivism imported")
    except Exception as e:
        print(f"✗ juvenile_recidivism: {e}")
        return False
    
    try:
        import victim_vulnerability
        print("✓ victim_vulnerability imported")
    except Exception as e:
        print(f"✗ victim_vulnerability: {e}")
        return False
    
    try:
        import institutional_stress
        print("✓ institutional_stress imported")
    except Exception as e:
        print(f"✗ institutional_stress: {e}")
        return False
    
    try:
        import utils
        print("✓ utils imported")
    except Exception as e:
        print(f"✗ utils: {e}")
        return False
    
    print("\n✅ All modules imported successfully")
    return True


def test_data_loading():
    """Test that data files can be loaded"""
    print("\n" + "=" * 80)
    print("TEST 2: Data Loading")
    print("=" * 80)
    
    try:
        from data_preparation import load_juvenile_data, load_victim_data, load_institutional_stress_data
        
        print("Loading juvenile data...")
        juvenile = load_juvenile_data()
        print(f"  ✓ Juvenile data loaded: {len(juvenile)} datasets")
        
        print("Loading victim data...")
        victim = load_victim_data()
        print(f"  ✓ Victim data loaded: {len(victim)} datasets")
        
        print("Loading institutional stress data...")
        stress = load_institutional_stress_data()
        print(f"  ✓ Institutional stress data loaded: {len(stress)} datasets")
        
        print("\n✅ All data files loaded successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        return False


def test_data_merging():
    """Test that data can be merged properly"""
    print("\n" + "=" * 80)
    print("TEST 3: Data Merging & Preparation")
    print("=" * 80)
    
    try:
        from data_preparation import (merge_juvenile_features, merge_victim_features, 
                                     merge_institutional_stress_features)
        
        print("Merging juvenile features...")
        juvenile_df = merge_juvenile_features()
        print(f"  ✓ Juvenile features merged: {juvenile_df.shape}")
        
        print("Merging victim features...")
        victim_df = merge_victim_features()
        print(f"  ✓ Victim features merged: {victim_df.shape}")
        
        print("Merging institutional stress features...")
        stress_df = merge_institutional_stress_features()
        print(f"  ✓ Institutional stress features merged: {stress_df.shape}")
        
        print("\n✅ All data merging successful")
        return True
        
    except Exception as e:
        print(f"\n✗ Data merging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test that models can be initialized"""
    print("\n" + "=" * 80)
    print("TEST 4: Model Initialization")
    print("=" * 80)
    
    try:
        from juvenile_recidivism import JuvenileRecidivismModel
        model1 = JuvenileRecidivismModel()
        print("✓ JuvenileRecidivismModel initialized")
    except Exception as e:
        print(f"✗ JuvenileRecidivismModel: {e}")
        return False
    
    try:
        from victim_vulnerability import VictimVulnerabilityAnalysis
        model2 = VictimVulnerabilityAnalysis()
        print("✓ VictimVulnerabilityAnalysis initialized")
    except Exception as e:
        print(f"✗ VictimVulnerabilityAnalysis: {e}")
        return False
    
    try:
        from institutional_stress import InstitutionalStressEWS
        model3 = InstitutionalStressEWS()
        print("✓ InstitutionalStressEWS initialized")
    except Exception as e:
        print(f"✗ InstitutionalStressEWS: {e}")
        return False
    
    print("\n✅ All models initialized successfully")
    return True


def test_feature_preparation():
    """Test that features can be prepared for modeling"""
    print("\n" + "=" * 80)
    print("TEST 5: Feature Preparation")
    print("=" * 80)
    
    try:
        from data_preparation import merge_juvenile_features
        from juvenile_recidivism import JuvenileRecidivismModel
        
        print("Preparing juvenile features...")
        df = merge_juvenile_features()
        model = JuvenileRecidivismModel()
        X, y, metadata, features = model.prepare_features(df)
        
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        print(f"  Features used: {len(features)}")
        
        if len(X) > 0:
            print("\n✅ Feature preparation successful")
            return True
        else:
            print("\n⚠️  Feature preparation resulted in empty dataset")
            return False
            
    except Exception as e:
        print(f"\n✗ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration settings"""
    print("\n" + "=" * 80)
    print("TEST 6: Configuration Validation")
    print("=" * 80)
    
    try:
        from config import (DATA_PATH, OUTPUT_PATH, RISK_THRESHOLDS, 
                           JUVENILE_RECIDIVISM_CONFIG, EWS_CONFIG)
        
        print(f"Data path: {DATA_PATH}")
        print(f"Output path: {OUTPUT_PATH}")
        print(f"Risk thresholds: {RISK_THRESHOLDS}")
        print(f"Juvenile config: {list(JUVENILE_RECIDIVISM_CONFIG.keys())}")
        print(f"EWS config: {list(EWS_CONFIG.keys())}")
        
        print("\n✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        return False


def run_all_tests():
    """Execute all tests"""
    print("\n" + "=" * 100)
    print("TESTING SUITE - Crime Data Analysis Project")
    print("=" * 100)
    
    results = {
        'Imports': test_imports(),
        'Data Loading': test_data_loading(),
        'Data Merging': test_data_merging(),
        'Model Initialization': test_model_initialization(),
        'Feature Preparation': test_feature_preparation(),
        'Configuration': test_configuration()
    }
    
    # Summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED - Project is ready to run!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
