from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)

def preprocess_data(df):
    processed_df = df.copy()
    
    # Handle TNM staging
    t_stage_order = {'T0': 0, 'Tis': 1, 'T1mi_a_b': 2, 'T1c': 3, 
                     'T2': 4, 'T3': 5, 'T4': 6, 'TX': 7}
    n_stage_order = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'NX': 4}
    m_stage_order = {'M0': 0, 'M1': 1, 'MX': 2}
    
    # Convert stages to ordinal
    processed_df['stage_t_ordinal'] = processed_df['stage_t'].map(t_stage_order)
    processed_df['stage_n_ordinal'] = processed_df['stage_n'].map(n_stage_order)
    processed_df['stage_m_ordinal'] = processed_df['stage_m'].map(m_stage_order)
    
    # Handle numerical features
    numeric_features = ['years_at_dx', 'years_to_survival_followup']
    numeric_imputer = SimpleImputer(strategy='median')
    processed_df[numeric_features] = numeric_imputer.fit_transform(processed_df[numeric_features])
    
    # Handle categorical features
    categorical_features = ['race', 'sex', 'subtype', 'is_adjuvant_rt_given']
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    processed_df[categorical_features] = categorical_imputer.fit_transform(processed_df[categorical_features])
    
    # Create binary event indicator
    processed_df['event_binary'] = (processed_df['event'] == 'death').astype(int)
    
    return processed_df

def prepare_model_data(processed_df):
    # Select features for modeling
    features = [
        'stage_t_ordinal', 'stage_n_ordinal', 'stage_m_ordinal',
        'years_at_dx', 'is_adjuvant_rt_given', 'race', 'sex', 'subtype'
    ]
    
    # Create dummy variables for categorical features
    categorical_features = ['race', 'sex', 'subtype', 'is_adjuvant_rt_given']
    model_df = pd.get_dummies(processed_df[features], columns=categorical_features)
    
    # Create the structured array for survival data
    dtype = [('event', bool), ('time', float)]
    y = np.zeros(len(processed_df), dtype=dtype)
    y['event'] = processed_df['event_binary'].astype(bool)
    y['time'] = processed_df['years_to_survival_followup']
    
    return model_df, y

def train_survival_model(model_df, y):
    # Split features and target
    X = model_df
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['event']
    )
    
    # Scale numerical features
    numerical_cols = ['stage_t_ordinal', 'stage_n_ordinal', 'stage_m_ordinal', 'years_at_dx']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train model
    model = RandomSurvivalForest(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    # Predict risk scores
    risk_scores = model.predict(X_test)
    
    # Calculate C-index
    c_index = model.score(X_test, y_test)
    
    # Calculate survival function
    survival_funcs = model.predict_survival_function(X_test)
    
    # Get unique time points from the survival functions
    time_points = [1, 2, 3, 5]
    metrics = {
        'c_index': c_index,
        'survival_probabilities': {}
    }
    
    # Get survival probabilities at different time points
    for t in time_points:
        # Find the closest time point in the survival function
        closest_times = [abs(sf.x - t) for sf in survival_funcs]
        closest_idx = np.argmin(closest_times)
        
        # Get the survival probabilities at that time
        surv_probs = np.array([sf.y[closest_idx] for sf in survival_funcs])
        metrics['survival_probabilities'][f't_{t}'] = surv_probs
    
    return metrics

def run_analysis():
    print_section("DATA LOADING AND PREPROCESSING")
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv('tcga_survival.csv')
    print(f"Initial dataset shape: {df.shape}")
    
    # Print initial data statistics
    print("\nInitial missing value counts:")
    print(df.isnull().sum())
    
    # Preprocess data
    print("\nPreprocessing data...")
    processed_df = preprocess_data(df)
    print("Preprocessing complete.")
    print(f"Processed dataset shape: {processed_df.shape}")
    
    print_section("MODEL PREPARATION AND TRAINING")
    
    # Prepare data for modeling
    print("\nPreparing features for modeling...")
    model_df, y = prepare_model_data(processed_df)
    print(f"Number of features after encoding: {model_df.shape[1]}")
    print("\nFeature names:")
    print(model_df.columns.tolist())
    
    # Train model
    print("\nTraining Random Survival Forest model...")
    model, X_train, X_test, y_train, y_test = train_survival_model(model_df, y)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    
    return model, processed_df, metrics

def visualize_results(model, processed_df, metrics):
    print_section("MODEL PERFORMANCE AND ANALYSIS")
    
    # Print model performance metrics
    print("\nModel Performance Metrics:")
    print(f"Overall C-index: {metrics['c_index']:.3f}")
    
    # Print survival probabilities
    print("\nSurvival Probability Analysis:")
    for time_point, probs in metrics['survival_probabilities'].items():
        mean_prob = np.mean(probs)
        median_prob = np.median(probs)
        print(f"\nTime point {time_point}:")
        print(f"  Mean survival probability: {mean_prob:.3f}")
        print(f"  Median survival probability: {median_prob:.3f}")
        print(f"  25th percentile: {np.percentile(probs, 25):.3f}")
        print(f"  75th percentile: {np.percentile(probs, 75):.3f}")
    
    print_section("DATASET ANALYSIS")
    
    # Overall dataset statistics
    print("\nOverall Dataset Statistics:")
    print(f"Total number of patients: {len(processed_df)}")
    print(f"Number of deaths: {processed_df['event_binary'].sum()}")
    print(f"Number of censored cases: {len(processed_df) - processed_df['event_binary'].sum()}")
    print(f"Overall death rate: {processed_df['event_binary'].mean():.2%}")
    
    # Cancer subtype analysis
    print("\nCancer Subtype Analysis:")
    subtype_stats = []
    for subtype in processed_df['subtype'].unique():
        mask = processed_df['subtype'] == subtype
        subtype_data = processed_df[mask]
        stats = {
            'Subtype': subtype,
            'Samples': len(subtype_data),
            'Deaths': subtype_data['event_binary'].sum(),
            'Death Rate': subtype_data['event_binary'].mean(),
            'Median Survival': subtype_data['years_to_survival_followup'].median(),
            'Mean Age': subtype_data['years_at_dx'].mean()
        }
        subtype_stats.append(stats)
    
    # Convert to DataFrame for nice printing
    stats_df = pd.DataFrame(subtype_stats)
    print("\nSubtype Statistics:")
    print(stats_df.to_string(float_format=lambda x: f"{x:.2f}"))
    
    print_section("DEMOGRAPHIC ANALYSIS")
    
    # Age distribution
    print("\nAge Statistics:")
    print(processed_df['years_at_dx'].describe())
    
    # Gender distribution
    print("\nGender Distribution:")
    print(processed_df['sex'].value_counts())
    
    # Race distribution
    print("\nRace Distribution:")
    print(processed_df['race'].value_counts())
    
    # Treatment analysis
    print("\nTreatment Analysis:")
    print("Adjuvant RT Distribution:")
    print(processed_df['is_adjuvant_rt_given'].value_counts())
    
    # Generate visualizations
    print_section("GENERATING VISUALIZATIONS")
    
    sns.set_style("darkgrid")
    
    # Create a figure with subplots for different visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # Survival curves by cancer subtype (2x2 grid)
    print("\nPlotting survival curves by cancer subtype...")
    gs = plt.GridSpec(2, 3)
    
    subtypes = processed_df['subtype'].unique()
    print(f"Found cancer subtypes: {subtypes}")
    
    for idx, subtype in enumerate(subtypes):
        if idx < 4:
            ax = plt.subplot(gs[idx//2, idx%2])
            mask = processed_df['subtype'] == subtype
            
            n_samples = mask.sum()
            n_events = processed_df.loc[mask, 'event_binary'].sum()
            
            kmf = KaplanMeierFitter()
            kmf.fit(
                processed_df.loc[mask, 'years_to_survival_followup'],
                processed_df.loc[mask, 'event_binary'],
                label=f"{subtype} (n={n_samples}, events={n_events})"
            )
            kmf.plot(ax=ax)
            ax.set_title(f'Survival Curve - {subtype}')
            ax.grid(True)
    
    # Distribution of survival times
    print("\nPlotting survival time distribution...")
    ax_time = plt.subplot(gs[:, 2])
    sns.histplot(
        data=processed_df,
        x='years_to_survival_followup',
        hue='subtype',
        multiple="stack",
        ax=ax_time
    )
    ax_time.set_title('Distribution of Survival Times by Subtype')
    ax_time.set_xlabel('Years to Follow-up')
    
    plt.tight_layout()
    plt.show()
    
    # Create additional plots for model insights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot event distribution
    print("\nPlotting event distribution...")
    event_counts = processed_df['event'].value_counts()
    sns.barplot(x=event_counts.index, y=event_counts.values, ax=ax1)
    ax1.set_title('Distribution of Events')
    ax1.set_ylabel('Count')
    
    # Plot age distribution by outcome
    print("\nPlotting age distribution...")
    sns.boxplot(
        data=processed_df,
        x='event',
        y='years_at_dx',
        ax=ax2
    )
    ax2.set_title('Age Distribution by Outcome')
    ax2.set_ylabel('Age at Diagnosis')
    
    plt.tight_layout()
    plt.show()
    
    # Print additional metrics
    print("\nDetailed Metrics:")
    print(f"Overall C-index: {metrics['c_index']:.3f}")
    
    # Print survival probabilities at different time points
    print("\nSurvival Probabilities Summary:")
    for time_point, probs in metrics['survival_probabilities'].items():
        mean_prob = np.mean(probs)
        print(f"Time point {time_point}: Mean survival probability = {mean_prob:.3f}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(processed_df)}")
    print(f"Number of events (deaths): {processed_df['event_binary'].sum()}")
    print(f"Censored samples: {len(processed_df) - processed_df['event_binary'].sum()}")
    
    # Print subtype-specific statistics
    print("\nSubtype-specific Statistics:")
    for subtype in processed_df['subtype'].unique():
        mask = processed_df['subtype'] == subtype
        n_samples = mask.sum()
        n_events = processed_df.loc[mask, 'event_binary'].sum()
        median_survival = processed_df.loc[mask, 'years_to_survival_followup'].median()
        print(f"\n{subtype}:")
        print(f"  Samples: {n_samples}")
        print(f"  Events: {n_events}")
        print(f"  Event rate: {n_events/n_samples:.2%}")
        print(f"  Median survival time: {median_survival:.2f} years")


if __name__ == "__main__":
    print_section("SURVIVAL ANALYSIS PROJECT")
    print("\nInitiating analysis pipeline...")
    
    # Run the analysis
    model, processed_df, metrics = run_analysis()
    
    # Generate visualizations and detailed analysis
    visualize_results(model, processed_df, metrics)
    
    print_section("ANALYSIS COMPLETE")