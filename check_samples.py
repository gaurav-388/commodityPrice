import pandas as pd

# Load the ENHANCED data
print("Loading ENHANCED dataset...")
df = pd.read_csv('Bengal_Prices_2014-25_enhanced.csv')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

# Filter for 2024-2025
df_filtered = df[(df['date'].dt.year >= 2024) & (df['date'].dt.year <= 2025)]

print('='*105)
print('                  ENHANCED DATA ANALYSIS FOR 2024-2025')
print('='*105)

# Show districts in 2024-2025
print("\n📍 DISTRICTS IN 2024-2025:")
print("-"*50)
districts_2024_25 = df_filtered['district'].unique()
for d in sorted(districts_2024_25):
    count = len(df_filtered[df_filtered['district'] == d])
    print(f"   • {d}: {count} records")

# Show commodities in 2024-2025
print("\n🌾 COMMODITIES IN 2024-2025:")
print("-"*50)
commodities_2024_25 = df_filtered['commodity_name'].unique()
for c in sorted(commodities_2024_25):
    count = len(df_filtered[df_filtered['commodity_name'] == c])
    print(f"   • {c}: {count} records")

# Show year-wise breakdown
print("\n📅 YEAR-WISE BREAKDOWN:")
print("-"*50)
for year in [2024, 2025]:
    year_data = df_filtered[df_filtered['date'].dt.year == year]
    print(f"   {year}: {len(year_data)} records")

# Get 100 random samples from 2024-2025
print("\n" + "="*105)
print('                     100 RANDOM SAMPLES FROM ENHANCED 2024-2025 DATA')
print("="*105)
header = f"{'#':<4} {'Date':<12} {'District':<22} {'Commodity':<12} {'Price (Rs/Qtl)':<15}"
print(header)
print('-'*105)

samples = df_filtered.sample(n=min(100, len(df_filtered)), random_state=456)

for i, (idx, row) in enumerate(samples.iterrows(), 1):
    date_str = row['date'].strftime('%Y-%m-%d')
    district = str(row['district'])[:20]
    commodity = str(row['commodity_name'])[:10]
    price = row['modal_price(rs)']
    print(f"{i:<4} {date_str:<12} {district:<22} {commodity:<12} {price:<15.2f}")

# Summary
print('-'*105)
print(f"Total records in 2024-2025: {len(df_filtered)}")
print(f"Unique Districts: {df_filtered['district'].nunique()}")
print(f"Unique Commodities: {df_filtered['commodity_name'].nunique()}")
print('='*105)
