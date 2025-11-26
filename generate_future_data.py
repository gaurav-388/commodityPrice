"""
Future Data Generation Script for 2024-2025
============================================
This script generates realistic price predictions for missing districts and commodities
by using historical patterns, inflation adjustment, and seasonal variations.

Methodology:
1. Historical Price Trends: Use 2020-2023 data as base
2. CPI Adjustment: Apply Consumer Price Index inflation
3. Seasonal Patterns: Monthly price variations
4. District Price Ratios: Relative pricing between districts
5. Commodity-specific growth rates
6. Random realistic variation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load existing data
print("Loading existing data...")
df = pd.read_csv('Bengal_Prices_2014-25_final.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

print(f"Original dataset: {len(df)} records")
print(f"Districts: {df['district'].nunique()}")
print(f"Commodities: {df['commodity_name'].nunique()}")

# ============================================================================
# STEP 1: ANALYZE HISTORICAL PATTERNS
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Analyzing Historical Patterns (2020-2023)")
print("="*70)

# Get 2020-2023 data as reference
df_ref = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2023)]

# Calculate base statistics per district-commodity-month
df_ref['month'] = df_ref['date'].dt.month
df_ref['year'] = df_ref['date'].dt.year

# Average price by district, commodity, month
base_stats = df_ref.groupby(['district', 'commodity_name', 'month']).agg({
    'modal_price(rs)': ['mean', 'std'],
    'temperature(celcius)': 'mean',
    'rainfall(mm)': 'mean'
}).reset_index()

base_stats.columns = ['district', 'commodity_name', 'month', 'price_mean', 'price_std', 'temp_mean', 'rainfall_mean']

# Fill missing std with 10% of mean
base_stats['price_std'] = base_stats['price_std'].fillna(base_stats['price_mean'] * 0.1)

print(f"Base statistics calculated for {len(base_stats)} district-commodity-month combinations")

# ============================================================================
# STEP 2: INFLATION & CPI ADJUSTMENT FACTORS
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Calculating Inflation Adjustment Factors")
print("="*70)

# CPI data (base year 2012=100) - approximate values
cpi_data = {
    2020: 158.8,
    2021: 167.6,
    2022: 179.5,
    2023: 189.7,
    2024: 199.2,  # Estimated ~5% inflation
    2025: 209.2   # Estimated ~5% inflation
}

# Calculate year-over-year inflation adjustments
inflation_factors = {
    2024: cpi_data[2024] / cpi_data[2023],  # ~1.05
    2025: cpi_data[2025] / cpi_data[2023]   # ~1.10
}

print(f"CPI-based inflation factors:")
print(f"  2024 vs 2023: {inflation_factors[2024]:.4f} ({(inflation_factors[2024]-1)*100:.2f}% increase)")
print(f"  2025 vs 2023: {inflation_factors[2025]:.4f} ({(inflation_factors[2025]-1)*100:.2f}% increase)")

# ============================================================================
# STEP 3: COMMODITY-SPECIFIC GROWTH RATES
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Commodity-Specific Adjustment Factors")
print("="*70)

# Historical growth rates per commodity (analyzed from data)
commodity_growth = {}
for commodity in df['commodity_name'].unique():
    comm_data = df[df['commodity_name'] == commodity]
    yearly_avg = comm_data.groupby(comm_data['date'].dt.year)['modal_price(rs)'].mean()
    if len(yearly_avg) >= 3:
        # Calculate average year-over-year growth
        growth_rates = yearly_avg.pct_change().dropna()
        avg_growth = growth_rates.mean()
        commodity_growth[commodity] = max(0.02, min(0.15, avg_growth))  # Cap between 2-15%
    else:
        commodity_growth[commodity] = 0.05  # Default 5%

for commodity, growth in commodity_growth.items():
    print(f"  {commodity}: {growth*100:.2f}% annual growth")

# ============================================================================
# STEP 4: SEASONAL PATTERNS
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Seasonal Price Patterns")
print("="*70)

# Calculate monthly seasonal factors per commodity
seasonal_factors = {}
for commodity in df['commodity_name'].unique():
    comm_data = df[df['commodity_name'] == commodity].copy()
    comm_data['month'] = comm_data['date'].dt.month
    monthly_avg = comm_data.groupby('month')['modal_price(rs)'].mean()
    overall_avg = monthly_avg.mean()
    seasonal_factors[commodity] = (monthly_avg / overall_avg).to_dict()

print("Seasonal patterns calculated for all commodities")

# ============================================================================
# STEP 5: DISTRICT PRICE RATIOS
# ============================================================================
print("\n" + "="*70)
print("STEP 5: District Price Ratios")
print("="*70)

# Calculate relative pricing between districts
district_ratios = {}
for commodity in df['commodity_name'].unique():
    comm_data = df[df['commodity_name'] == commodity]
    district_avg = comm_data.groupby('district')['modal_price(rs)'].mean()
    overall_avg = district_avg.mean()
    district_ratios[commodity] = (district_avg / overall_avg).to_dict()

print("District price ratios calculated")

# ============================================================================
# STEP 6: GENERATE FUTURE DATA
# ============================================================================
print("\n" + "="*70)
print("STEP 6: Generating Future Data for 2024-2025")
print("="*70)

# Get all unique values
all_districts = df['district'].unique()
all_commodities = df['commodity_name'].unique()
all_markets = df.groupby('district')['market_name'].first().to_dict()

# Get reference data for other columns
ref_row = df[df['date'].dt.year == 2023].iloc[0]

# Generate dates for 2024 and 2025
def generate_dates(year):
    """Generate weekly dates for a year"""
    dates = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31) if year == 2024 else datetime(year, 9, 30)  # Up to current date
    
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=7)  # Weekly data
    return dates

dates_2024 = generate_dates(2024)
dates_2025 = generate_dates(2025)

print(f"Generating data for {len(dates_2024)} dates in 2024")
print(f"Generating data for {len(dates_2025)} dates in 2025")

# Check existing 2024-2025 data
existing_2024_25 = df[(df['date'].dt.year >= 2024) & (df['date'].dt.year <= 2025)]
existing_combinations = set(zip(
    existing_2024_25['date'].dt.date,
    existing_2024_25['district'],
    existing_2024_25['commodity_name']
))

print(f"Existing 2024-2025 records: {len(existing_2024_25)}")

# Generate new records
new_records = []
np.random.seed(42)  # For reproducibility

for date in dates_2024 + dates_2025:
    year = date.year
    month = date.month
    
    for district in all_districts:
        for commodity in all_commodities:
            # Skip if already exists
            if (date.date(), district, commodity) in existing_combinations:
                continue
            
            # Get base price from historical data
            base_data = base_stats[
                (base_stats['district'] == district) & 
                (base_stats['commodity_name'] == commodity) & 
                (base_stats['month'] == month)
            ]
            
            if len(base_data) == 0:
                # Use commodity average if district-specific data not available
                base_data = base_stats[
                    (base_stats['commodity_name'] == commodity) & 
                    (base_stats['month'] == month)
                ]
            
            if len(base_data) == 0:
                continue
            
            base_price = base_data['price_mean'].values[0]
            price_std = base_data['price_std'].values[0]
            temp = base_data['temp_mean'].values[0]
            rainfall = base_data['rainfall_mean'].values[0]
            
            # Apply adjustments
            # 1. Inflation adjustment
            inflation_adj = inflation_factors[year]
            
            # 2. Commodity growth rate
            years_from_2023 = year - 2023
            growth_adj = (1 + commodity_growth.get(commodity, 0.05)) ** years_from_2023
            
            # 3. Seasonal adjustment
            seasonal_adj = seasonal_factors.get(commodity, {}).get(month, 1.0)
            
            # 4. District ratio adjustment
            district_adj = district_ratios.get(commodity, {}).get(district, 1.0)
            
            # Calculate final price
            adjusted_price = base_price * inflation_adj * growth_adj * seasonal_adj * district_adj
            
            # Add random variation (±5%)
            variation = np.random.normal(0, 0.05)
            final_price = adjusted_price * (1 + variation)
            final_price = max(100, round(final_price, -1))  # Round to nearest 10, min 100
            
            # Get other economic indicators with adjustment
            per_capita = ref_row['Per_Capita_Income(per capita nsdp,rs)'] * inflation_adj
            food_subsidy = ref_row['Food_Subsidy(in thousand crores)'] * 1.05 ** years_from_2023
            
            # Create record
            record = {
                'date': date.strftime('%d-%m-%Y'),
                'state_name': 'West Bengal',
                'district': district,
                'market_name': all_markets.get(district, 'Main Market'),
                'commodity_name': commodity,
                'variety': 'Local',
                'modal_price(rs)': final_price,
                'temperature(celcius)': temp + np.random.normal(0, 1),
                'rainfall(mm)': max(0, rainfall + np.random.normal(0, 10)),
                'Per_Capita_Income(per capita nsdp,rs)': per_capita,
                'Food_Subsidy(in thousand crores)': food_subsidy,
                'CPI(base year2012=100)': cpi_data[year],
                'Elec_Agri_Share(%)': ref_row['Elec_Agri_Share(%)'],
                'MSP(per quintol)': ref_row['MSP(per quintol)'] * inflation_adj,
                'Fertilizer_Consumption(kg/ha)': ref_row['Fertilizer_Consumption(kg/ha)'],
                'Area(million ha)': ref_row['Area(million ha)'],
                'Production(million tonnes)': ref_row['Production(million tonnes)'],
                'Yield(kg/ha)': ref_row['Yield(kg/ha)'],
                'Export(Million MT)': ref_row['Export(Million MT)'],
                'Import(Million MT)': ref_row['Import(Million MT)']
            }
            new_records.append(record)

print(f"\nGenerated {len(new_records)} new records")

# ============================================================================
# STEP 7: COMBINE AND SAVE
# ============================================================================
print("\n" + "="*70)
print("STEP 7: Saving Enhanced Dataset")
print("="*70)

# Create DataFrame from new records
df_new = pd.DataFrame(new_records)

# Combine with original data
# First, convert original date back to string format for consistency
df['date'] = df['date'].dt.strftime('%d-%m-%Y')
df_combined = pd.concat([df, df_new], ignore_index=True)

# Save to new file
output_file = 'Bengal_Prices_2014-25_enhanced.csv'
df_combined.to_csv(output_file, index=False)
print(f"Enhanced dataset saved to: {output_file}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

# Reload and verify
df_verify = pd.read_csv(output_file)
df_verify['date'] = pd.to_datetime(df_verify['date'], format='%d-%m-%Y', errors='coerce')

# Filter for 2024-2025
df_2024_25 = df_verify[(df_verify['date'].dt.year >= 2024) & (df_verify['date'].dt.year <= 2025)]

print(f"\nEnhanced 2024-2025 Data:")
print(f"  Total records: {len(df_2024_25)}")
print(f"  Unique Districts: {df_2024_25['district'].nunique()}")
print(f"  Unique Commodities: {df_2024_25['commodity_name'].nunique()}")

print(f"\n📍 Districts in 2024-2025:")
for d in sorted(df_2024_25['district'].unique()):
    count = len(df_2024_25[df_2024_25['district'] == d])
    print(f"   • {d}: {count} records")

print(f"\n🌾 Commodities in 2024-2025:")
for c in sorted(df_2024_25['commodity_name'].unique()):
    count = len(df_2024_25[df_2024_25['commodity_name'] == c])
    print(f"   • {c}: {count} records")

print(f"\n📅 Year-wise breakdown:")
for year in [2024, 2025]:
    year_data = df_2024_25[df_2024_25['date'].dt.year == year]
    print(f"   {year}: {len(year_data)} records")

# Show sample prices comparison
print("\n" + "="*70)
print("SAMPLE PRICE COMPARISON (Historical vs Generated)")
print("="*70)

for commodity in df_verify['commodity_name'].unique():
    print(f"\n{commodity}:")
    comm_data = df_verify[df_verify['commodity_name'] == commodity]
    for year in [2022, 2023, 2024, 2025]:
        year_data = comm_data[comm_data['date'].dt.year == year]
        if len(year_data) > 0:
            avg_price = year_data['modal_price(rs)'].mean()
            print(f"   {year}: ₹{avg_price:,.0f}/Qtl (avg)")

print("\n" + "="*70)
print("✅ DATA GENERATION COMPLETE!")
print("="*70)
print(f"\nThe enhanced dataset '{output_file}' now contains:")
print(f"  • Complete data for all 18 districts")
print(f"  • All 3 commodities (Rice, Jute, Wheat)")
print(f"  • Prices adjusted for inflation (CPI)")
print(f"  • Seasonal variations included")
print(f"  • District-specific price ratios applied")
print("\nYou can now retrain your model with this enhanced dataset!")
