"""
Update Database and Model Configuration
========================================
This script updates the SQLite database with enhanced 2024-2025 data
and ensures the app uses the newly trained model.
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime
import pickle
import json

print("="*80)
print("UPDATING DATABASE AND MODEL CONFIGURATION")
print("="*80)

# ============================================================================
# STEP 1: UPDATE DATABASE WITH ENHANCED DATA
# ============================================================================
print("\n📂 STEP 1: Updating SQLite Database...")

db_path = 'commodity_prices.db'
csv_path = 'Bengal_Prices_2014-25_enhanced.csv'

# Backup existing database
if os.path.exists(db_path):
    backup_path = db_path + '.backup'
    os.rename(db_path, backup_path)
    print(f"   Backed up existing database to: {backup_path}")

# Load enhanced CSV
print(f"   Loading enhanced CSV: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   Loaded {len(df):,} records")

# Convert date format
print("   Converting date format...")
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

# Create new database
print(f"   Creating new database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create main table
print("   Creating prices table...")
df.to_sql('prices', conn, if_exists='replace', index=False)

# Create indexes
print("   Creating indexes...")
cursor.execute('CREATE INDEX idx_district ON prices(district)')
cursor.execute('CREATE INDEX idx_market ON prices(market_name)')
cursor.execute('CREATE INDEX idx_commodity ON prices(commodity_name)')
cursor.execute('CREATE INDEX idx_variety ON prices(variety)')
cursor.execute('CREATE INDEX idx_date ON prices(date_str)')
cursor.execute('CREATE INDEX idx_district_market ON prices(district, market_name)')
cursor.execute('CREATE INDEX idx_commodity_variety ON prices(commodity_name, variety)')

# Create metadata table
print("   Creating metadata table...")
cursor.execute('''
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
''')
cursor.execute("INSERT INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))
cursor.execute("INSERT INTO metadata VALUES ('total_records', ?)", (str(len(df)),))
cursor.execute("INSERT INTO metadata VALUES ('csv_file', ?)", (csv_path,))
cursor.execute("INSERT INTO metadata VALUES ('enhanced_data', 'true')")

# Create lookup tables
print("   Creating lookup tables...")

# Districts lookup
districts = df['district'].unique()
cursor.execute('CREATE TABLE lookup_districts (district TEXT PRIMARY KEY)')
cursor.executemany('INSERT INTO lookup_districts VALUES (?)', [(d,) for d in sorted(districts)])

# Markets by district lookup
cursor.execute('''
    CREATE TABLE lookup_markets (
        district TEXT,
        market_name TEXT,
        PRIMARY KEY (district, market_name)
    )
''')
markets_by_district = df[['district', 'market_name']].drop_duplicates()
cursor.executemany(
    'INSERT INTO lookup_markets VALUES (?, ?)',
    markets_by_district.values.tolist()
)

# Commodities lookup
commodities = df['commodity_name'].unique()
cursor.execute('CREATE TABLE lookup_commodities (commodity_name TEXT PRIMARY KEY)')
cursor.executemany('INSERT INTO lookup_commodities VALUES (?)', [(c,) for c in sorted(commodities)])

# Varieties by commodity lookup
cursor.execute('''
    CREATE TABLE lookup_varieties (
        commodity_name TEXT,
        variety TEXT,
        PRIMARY KEY (commodity_name, variety)
    )
''')
varieties_by_commodity = df[['commodity_name', 'variety']].drop_duplicates()
cursor.executemany(
    'INSERT INTO lookup_varieties VALUES (?, ?)',
    varieties_by_commodity.values.tolist()
)

conn.commit()

# Verify database
print("\n   Verification:")
cursor.execute("SELECT COUNT(*) FROM prices")
count = cursor.fetchone()[0]
print(f"   • Total records: {count:,}")

cursor.execute("SELECT COUNT(*) FROM lookup_districts")
districts_count = cursor.fetchone()[0]
print(f"   • Districts: {districts_count}")

cursor.execute("SELECT COUNT(*) FROM lookup_commodities")
commodities_count = cursor.fetchone()[0]
print(f"   • Commodities: {commodities_count}")

conn.close()
print("   ✅ Database updated successfully!")

# ============================================================================
# STEP 2: UPDATE MODEL FILE FOR COMPATIBILITY
# ============================================================================
print("\n🔧 STEP 2: Updating Model Configuration...")

# Load the new model and encoders
with open('models/xgboost_final_model.pkl', 'rb') as f:
    new_model = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    new_label_encoders = pickle.load(f)

with open('models/feature_columns.json', 'r') as f:
    new_features = json.load(f)

# Create model_data dict compatible with existing app
model_data = {
    'model': new_model,
    'features': new_features,
    'label_encoders': new_label_encoders,
    'variety_mapping': {},  # Add empty variety mapping
    'training_date': datetime.now().isoformat(),
    'dataset': 'Bengal_Prices_2014-25_enhanced.csv'
}

# Save as improved model (the filename app.py expects)
with open('models/xgboost_improved_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("   ✅ Saved: models/xgboost_improved_model.pkl")

# ============================================================================
# STEP 3: VERIFY 2024-2025 DATA IN DATABASE
# ============================================================================
print("\n📊 STEP 3: Verifying 2024-2025 Data in Database...")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check 2024-2025 data
cursor.execute("""
    SELECT 
        strftime('%Y', date_str) as year,
        COUNT(*) as records,
        COUNT(DISTINCT district) as districts,
        COUNT(DISTINCT commodity_name) as commodities
    FROM prices 
    WHERE strftime('%Y', date_str) IN ('2024', '2025')
    GROUP BY strftime('%Y', date_str)
""")
results = cursor.fetchall()

print("\n   Year-wise summary:")
for row in results:
    print(f"   • {row[0]}: {row[1]:,} records, {row[2]} districts, {row[3]} commodities")

# Sample data from 2025
cursor.execute("""
    SELECT district, commodity_name, "modal_price(rs)", date_str
    FROM prices 
    WHERE strftime('%Y', date_str) = '2025'
    ORDER BY RANDOM()
    LIMIT 10
""")
samples = cursor.fetchall()

print("\n   Sample 2025 records:")
for row in samples:
    print(f"   • {row[3]} | {row[0][:15]:<15} | {row[1]:<8} | ₹{row[2]:,.0f}")

conn.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("                    ✅ UPDATE COMPLETE!")
print("="*80)
print(f"""
Database Updated:
-----------------
• File: {db_path}
• Records: {count:,}
• Districts: {districts_count}
• Commodities: {commodities_count}
• Size: {os.path.getsize(db_path) / (1024*1024):.2f} MB

Model Updated:
--------------
• File: models/xgboost_improved_model.pkl
• Features: {len(new_features)}

Next Step:
----------
Run: python app_stable.py
Or:  python run_server.py

The server will now have complete 2024-2025 predictions!
""")
print("="*80)
