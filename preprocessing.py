"""
Data Preprocessing Module for Commodity Price Prediction
Handles data cleaning, feature engineering, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, data_path):
        """Initialize preprocessor with data path"""
        self.data_path = data_path
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'modal_price(rs)'
        
    def load_data(self):
        """Load the dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def clean_data(self):
        """Clean and handle missing values"""
        print("\nCleaning data...")
        initial_rows = len(self.df)
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d-%m-%Y')
        
        # Drop rows with missing target values
        self.df = self.df.dropna(subset=[self.target_column])
        
        # Fill missing values for numeric columns with median
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill missing values for categorical columns with mode
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'date' and self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print(f"Rows after cleaning: {len(self.df)} (removed {initial_rows - len(self.df)} rows)")
        return self.df
    
    def engineer_features(self):
        """Create new features from existing data"""
        print("\nEngineering features...")
        
        # Time-based features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        
        # Lag features for price (previous values)
        self.df = self.df.sort_values('date')
        for lag in [7, 14, 30, 90]:
            self.df[f'price_lag_{lag}'] = self.df.groupby(['commodity_name', 'district'])[self.target_column].shift(lag)
        
        # Rolling statistics
        for window in [7, 30, 90]:
            self.df[f'price_rolling_mean_{window}'] = self.df.groupby(['commodity_name', 'district'])[self.target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.df[f'price_rolling_std_{window}'] = self.df.groupby(['commodity_name', 'district'])[self.target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Price change features
        self.df['price_change_7d'] = self.df.groupby(['commodity_name', 'district'])[self.target_column].pct_change(7)
        self.df['price_change_30d'] = self.df.groupby(['commodity_name', 'district'])[self.target_column].pct_change(30)
        
        # Interaction features
        self.df['temp_rainfall_interaction'] = self.df['temperature(celcius)'] * self.df['rainfall(mm)']
        self.df['price_to_msp_ratio'] = self.df[self.target_column] / (self.df['MSP(per quintol)'] + 1)
        self.df['production_per_area'] = self.df['Production(million tonnes)'] / (self.df['Area(million ha)'] + 0.001)
        
        # Seasonal indicators
        self.df['is_monsoon'] = self.df['month'].isin([6, 7, 8, 9]).astype(int)
        self.df['is_winter'] = self.df['month'].isin([11, 12, 1, 2]).astype(int)
        self.df['is_summer'] = self.df['month'].isin([3, 4, 5]).astype(int)
        
        # Fill any NaN values created by lag/rolling features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        print(f"Features after engineering: {self.df.shape[1]} columns")
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        categorical_columns = ['state_name', 'district', 'market_name', 
                              'commodity_name', 'variety']
        
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} unique values")
        
        return self.df
    
    def prepare_features(self):
        """Prepare final feature set for training"""
        print("\nPreparing features for training...")
        
        # Define features to exclude
        exclude_cols = ['date', 'state_name', 'district', 'market_name', 
                       'commodity_name', 'variety', self.target_column]
        
        # Select all columns except excluded ones
        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        print(f"Total features: {len(self.feature_columns)}")
        print(f"Feature names: {self.feature_columns[:10]}... (showing first 10)")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def run_pipeline(self, test_size=0.2):
        """Run the complete preprocessing pipeline"""
        print("="*60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.encode_categorical()
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return X_train, X_test, y_train, y_test, self.feature_columns


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor('Bengal_Prices_2014-25_final.csv')
    X_train, X_test, y_train, y_test, features = preprocessor.run_pipeline()
    
    print("\nPreprocessing Summary:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
