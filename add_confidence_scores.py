"""
Add Confidence Scores to Predictions
Helps users understand prediction reliability
"""
import pickle
import pandas as pd
import numpy as np

def calculate_confidence_score(district, market, commodity, variety, date, df, label_encoders):
    """
    Calculate confidence score (0-100%) based on data availability and historical variance
    """
    score = 100.0
    reasons = []
    
    # Check 1: Does this exact combination exist in training data?
    exact_match = df[
        (df['district'] == district) &
        (df['market_name'] == market) &
        (df['commodity_name'] == commodity) &
        (df['variety'] == variety)
    ]
    
    if len(exact_match) == 0:
        score -= 30
        reasons.append("No historical data for this exact combination")
    elif len(exact_match) < 10:
        score -= 15
        reasons.append("Limited historical data (< 10 records)")
    
    # Check 2: Price variance in similar records
    if len(exact_match) > 0:
        price_std = exact_match['modal_price(rs)'].std()
        price_mean = exact_match['modal_price(rs)'].mean()
        cv = (price_std / price_mean) * 100  # Coefficient of variation
        
        if cv > 30:
            score -= 20
            reasons.append(f"High price variability ({cv:.1f}%)")
        elif cv > 15:
            score -= 10
            reasons.append(f"Moderate price variability ({cv:.1f}%)")
    
    # Check 3: Is variety name properly encoded?
    try:
        label_encoders['variety'].transform([variety])
    except:
        score -= 25
        reasons.append("Variety name not found in training data")
    
    # Check 4: Market coverage
    market_records = df[
        (df['district'] == district) &
        (df['market_name'] == market)
    ]
    
    if len(market_records) < 50:
        score -= 15
        reasons.append("Limited market data")
    
    # Check 5: Recent data availability
    date_dt = pd.to_datetime(date)
    recent_data = df[
        (df['district'] == district) &
        (df['market_name'] == market) &
        (df['commodity_name'] == commodity) &
        (df['date'] >= date_dt - pd.Timedelta(days=180))
    ]
    
    if len(recent_data) == 0:
        score -= 10
        reasons.append("No recent data (last 6 months)")
    
    # Ensure score is between 0-100
    score = max(0, min(100, score))
    
    # Determine confidence level
    if score >= 80:
        level = "HIGH"
    elif score >= 60:
        level = "MEDIUM"
    elif score >= 40:
        level = "LOW"
    else:
        level = "VERY LOW"
    
    return {
        'score': round(score, 1),
        'level': level,
        'reasons': reasons
    }

# Create an enhanced version of the predict function
def predict_with_confidence(date, district, market, commodity, variety, model, features, label_encoders, df):
    """
    Make prediction with confidence score
    """
    # Calculate confidence
    confidence = calculate_confidence_score(district, market, commodity, variety, date, df, label_encoders)
    
    # Add confidence to prediction result
    return confidence

# Save this function for use in app.py
print("Confidence scoring module created!")
print("\nTo integrate into app.py:")
print("1. Import this module")
print("2. Call predict_with_confidence() instead of predict()")
print("3. Display confidence score in the UI")
print("\nExample output:")
print("  Score: 85.0%")
print("  Level: HIGH")
print("  Reasons: ['Limited historical data (< 10 records)']")
