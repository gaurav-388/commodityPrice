"""
Initialize SQLite database from CSV for faster and more reliable data access
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime

def init_database():
    """Initialize SQLite database from CSV file"""
    
    print("="*80)
    print("INITIALIZING SQLITE DATABASE")
    print("="*80)
    
    db_path = 'commodity_prices.db'
    csv_path = 'Bengal_Prices_2014-25_final.csv'
    
    # Check if database already exists
    if os.path.exists(db_path):
        print(f"Database already exists at: {db_path}")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping database initialization.")
            return
        os.remove(db_path)
        print(f"Deleted existing database.")
    
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Convert date format
    print("\nConverting date format...")
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Create database connection
    print(f"\nCreating database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Create main table
    print("Creating prices table...")
    df.to_sql('prices', conn, if_exists='replace', index=False)
    
    # Create indexes for faster queries
    print("Creating indexes...")
    cursor = conn.cursor()
    
    cursor.execute('CREATE INDEX idx_district ON prices(district)')
    cursor.execute('CREATE INDEX idx_market ON prices(market_name)')
    cursor.execute('CREATE INDEX idx_commodity ON prices(commodity_name)')
    cursor.execute('CREATE INDEX idx_variety ON prices(variety)')
    cursor.execute('CREATE INDEX idx_date ON prices(date_str)')
    cursor.execute('CREATE INDEX idx_district_market ON prices(district, market_name)')
    cursor.execute('CREATE INDEX idx_commodity_variety ON prices(commodity_name, variety)')
    
    # Create metadata table
    print("Creating metadata table...")
    cursor.execute('''
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    cursor.execute("INSERT INTO metadata VALUES ('created_at', ?)", (datetime.now().isoformat(),))
    cursor.execute("INSERT INTO metadata VALUES ('total_records', ?)", (str(len(df)),))
    cursor.execute("INSERT INTO metadata VALUES ('csv_file', ?)", (csv_path,))
    
    # Create lookup tables for faster dropdown queries
    print("Creating lookup tables...")
    
    # Districts lookup
    districts = df['district'].unique()
    cursor.execute('CREATE TABLE lookup_districts (district TEXT PRIMARY KEY)')
    cursor.executemany('INSERT INTO lookup_districts VALUES (?)', [(d,) for d in districts])
    
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
    cursor.executemany('INSERT INTO lookup_commodities VALUES (?)', [(c,) for c in commodities])
    
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
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    cursor.execute("SELECT COUNT(*) FROM prices")
    count = cursor.fetchone()[0]
    print(f"Total records in prices table: {count:,}")
    
    cursor.execute("SELECT COUNT(*) FROM lookup_districts")
    districts_count = cursor.fetchone()[0]
    print(f"Total districts: {districts_count}")
    
    cursor.execute("SELECT COUNT(*) FROM lookup_commodities")
    commodities_count = cursor.fetchone()[0]
    print(f"Total commodities: {commodities_count}")
    
    cursor.execute("SELECT COUNT(*) FROM lookup_markets")
    markets_count = cursor.fetchone()[0]
    print(f"Total market-district combinations: {markets_count}")
    
    cursor.execute("SELECT COUNT(*) FROM lookup_varieties")
    varieties_count = cursor.fetchone()[0]
    print(f"Total commodity-variety combinations: {varieties_count}")
    
    # Test query
    print("\nTesting sample query...")
    cursor.execute("""
        SELECT district, market_name, commodity_name, variety, "modal_price(rs)"
        FROM prices 
        LIMIT 5
    """)
    results = cursor.fetchall()
    print("\nSample records:")
    for row in results:
        print(f"  {row}")
    
    conn.close()
    
    print("\n" + "="*80)
    print("DATABASE INITIALIZATION COMPLETE!")
    print("="*80)
    print(f"Database file: {db_path}")
    print(f"Size: {os.path.getsize(db_path) / (1024*1024):.2f} MB")
    print("\nYou can now use this database in app.py for faster data access.")
    print("="*80)

if __name__ == '__main__':
    init_database()
