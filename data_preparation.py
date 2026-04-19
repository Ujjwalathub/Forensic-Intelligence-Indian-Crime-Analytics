"""
Data Preparation Module
Handles loading, cleaning, and standardizing Indian Crime Dataset
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = r'e:\Project\Data'


def clean_keys(df, state_col='Area_Name', district_col=None):
    """
    Standardize state and district names by converting to uppercase and stripping whitespace
    """
    if state_col in df.columns:
        df[state_col] = df[state_col].str.upper().str.strip()
    
    if district_col and district_col in df.columns:
        df[district_col] = df[district_col].str.upper().str.strip()
    
    return df


def load_juvenile_data():
    """
    Load juvenile recidivism data (Section XIII)
    Returns: Dict with education, economic, family, and recidivism dataframes
    """
    print("Loading Juvenile Data (Section XIII)...")
    
    juvenile_education = pd.read_csv(f'{DATA_PATH}/18_01_Juveniles_arrested_Education.csv')
    juvenile_economic = pd.read_csv(f'{DATA_PATH}/18_02_Juveniles_arrested_Economic_setup.csv')
    juvenile_family = pd.read_csv(f'{DATA_PATH}/18_03_Juveniles_arrested_Family_background.csv')
    juvenile_recidivism = pd.read_csv(f'{DATA_PATH}/18_04_Juveniles_arrested_Recidivism.csv')
    
    # Standardize keys
    for df in [juvenile_education, juvenile_economic, juvenile_family, juvenile_recidivism]:
        clean_keys(df, state_col='Area_Name')
    
    return {
        'education': juvenile_education,
        'economic': juvenile_economic,
        'family': juvenile_family,
        'recidivism': juvenile_recidivism
    }


def load_victim_data():
    """
    Load victim demographic data (Sections XV, XXI: 5.1 & 5.2)
    Returns: Dict with murder, culpable homicide, and rape victim dataframes
    """
    print("Loading Victim Data (Section XV & XXI)...")
    
    murder_victims = pd.read_csv(f'{DATA_PATH}/32_Murder_victim_age_sex.csv')
    ch_victims = pd.read_csv(f'{DATA_PATH}/33_CH_not_murder_victim_age_sex.csv')
    rape_victims = pd.read_csv(f'{DATA_PATH}/20_Victims_of_rape.csv')
    
    # Standardize keys
    for df in [murder_victims, ch_victims, rape_victims]:
        clean_keys(df, state_col='Area_Name')
    
    return {
        'murder': murder_victims,
        'culpable_homicide': ch_victims,
        'rape': rape_victims
    }


def load_institutional_stress_data():
    """
    Load complaints and human rights violation data (Section XIX, XXI: 6 & 11)
    Returns: Dict with complaints and custodial death dataframes
    """
    print("Loading Institutional Stress Data (Section XIX & XXI)...")
    
    complaints = pd.read_csv(f'{DATA_PATH}/25_Complaints_against_police.csv')
    
    # Load custodial death data
    custodial_death_remanded = pd.read_csv(f'{DATA_PATH}/40_01_Custodial_death_person_remanded.csv')
    custodial_death_not_remanded = pd.read_csv(f'{DATA_PATH}/40_02_Custodial_death_person_not_remanded.csv')
    custodial_death_production = pd.read_csv(f'{DATA_PATH}/40_03_Custodial_death_during_production.csv')
    custodial_death_hospitalization = pd.read_csv(f'{DATA_PATH}/40_04_Custodial_death_during_hospitalization_or_treatment.csv')
    custodial_death_others = pd.read_csv(f'{DATA_PATH}/40_05_Custodial_death_others.csv')
    
    # Load human rights violations
    hr_violations = pd.read_csv(f'{DATA_PATH}/35_Human_rights_violation_by_police.csv')
    
    # Standardize keys
    for df in [complaints, custodial_death_remanded, custodial_death_not_remanded, 
               custodial_death_production, custodial_death_hospitalization, 
               custodial_death_others, hr_violations]:
        clean_keys(df, state_col='Area_Name')
    
    return {
        'complaints': complaints,
        'custodial_remanded': custodial_death_remanded,
        'custodial_not_remanded': custodial_death_not_remanded,
        'custodial_production': custodial_death_production,
        'custodial_hospitalization': custodial_death_hospitalization,
        'custodial_others': custodial_death_others,
        'hr_violations': hr_violations
    }


def merge_juvenile_features():
    """
    Merge juvenile education, economic, family, and recidivism data
    Returns: Merged dataframe
    """
    print("\nMerging juvenile features...")
    
    juvenile_data = load_juvenile_data()
    
    # Extract numeric columns (education)
    education_cols = [col for col in juvenile_data['education'].columns if col.startswith('Education_')]
    education_features = juvenile_data['education'][['Area_Name', 'Year'] + education_cols].copy()
    
    # Extract numeric columns (economic)
    economic_cols = [col for col in juvenile_data['economic'].columns if col.startswith('Economic_')]
    economic_features = juvenile_data['economic'][['Area_Name', 'Year'] + economic_cols].copy()
    
    # Extract numeric columns (family)
    family_cols = [col for col in juvenile_data['family'].columns if col.startswith('Family_')]
    family_features = juvenile_data['family'][['Area_Name', 'Year'] + family_cols].copy()
    
    # Extract recidivism columns
    recidivism_cols = [col for col in juvenile_data['recidivism'].columns if col.startswith('Recidivism_')]
    recidivism_features = juvenile_data['recidivism'][['Area_Name', 'Year'] + recidivism_cols].copy()
    
    # Merge on Area_Name and Year
    merged = education_features.merge(economic_features, on=['Area_Name', 'Year'], how='outer')
    merged = merged.merge(family_features, on=['Area_Name', 'Year'], how='outer')
    merged = merged.merge(recidivism_features, on=['Area_Name', 'Year'], how='outer')
    
    # Fill NaN with 0
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    print(f"✓ Merged juvenile data shape: {merged.shape}")
    return merged


def merge_victim_features():
    """
    Merge victim data from multiple crime types
    Returns: Merged dataframe with victim statistics
    """
    print("\nMerging victim features...")
    
    victim_data = load_victim_data()
    
    # Extract victim columns
    murder_cols = [col for col in victim_data['murder'].columns if col.startswith('Victims_')]
    murder_features = victim_data['murder'][['Area_Name', 'Year', 'Group_Name'] + murder_cols].copy()
    
    ch_cols = [col for col in victim_data['culpable_homicide'].columns if col.startswith('Victims_')]
    ch_features = victim_data['culpable_homicide'][['Area_Name', 'Year', 'Group_Name'] + ch_cols].copy()
    
    # Combine murder and culpable homicide
    merged = pd.concat([murder_features, ch_features], ignore_index=True)
    
    # Fill NaN with 0
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    print(f"✓ Merged victim data shape: {merged.shape}")
    return merged


def merge_institutional_stress_features():
    """
    Merge complaints and custodial death data
    Returns: Merged dataframe with stress indicators
    """
    print("\nMerging institutional stress features...")
    
    stress_data = load_institutional_stress_data()
    
    # Extract complaint columns
    complaint_cols = [col for col in stress_data['complaints'].columns 
                     if col.startswith('CPA_') or col.startswith('CPB_') or col.startswith('CPC_')]
    complaints_features = stress_data['complaints'][['Area_Name', 'Year'] + complaint_cols].copy()
    
    # Aggregate custodial deaths
    custodial_data = pd.DataFrame()
    for key in ['custodial_remanded', 'custodial_not_remanded', 'custodial_production', 
                'custodial_hospitalization', 'custodial_others']:
        if key in stress_data and not stress_data[key].empty:
            custodial_data = pd.concat([custodial_data, stress_data[key]], ignore_index=True)
    
    # Extract HR violation columns
    hr_cols = [col for col in stress_data['hr_violations'].columns if col not in ['Area_Name', 'Year', 'Sub_Group_Name']]
    hr_features = stress_data['hr_violations'][['Area_Name', 'Year'] + hr_cols].copy()
    
    # Merge complaints with HR violations
    merged = complaints_features.merge(hr_features, on=['Area_Name', 'Year'], how='outer')
    
    # Fill NaN with 0
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    
    print(f"✓ Merged institutional stress data shape: {merged.shape}")
    return merged


if __name__ == "__main__":
    print("=" * 80)
    print("DATA PREPARATION MODULE")
    print("=" * 80)
    
    # Test loading and merging
    juvenile_df = merge_juvenile_features()
    print(f"\nJuvenile DataFrame shape: {juvenile_df.shape}")
    print(f"Columns: {juvenile_df.columns.tolist()[:10]}...")
    
    victim_df = merge_victim_features()
    print(f"\nVictim DataFrame shape: {victim_df.shape}")
    print(f"Columns: {victim_df.columns.tolist()[:10]}...")
    
    stress_df = merge_institutional_stress_features()
    print(f"\nInstitutional Stress DataFrame shape: {stress_df.shape}")
    print(f"Columns: {stress_df.columns.tolist()[:10]}...")
