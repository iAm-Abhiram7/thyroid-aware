import psycopg2
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

# Test 1: Direct psycopg2 connection
def test_psycopg2_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="thyroid_aware",
            user="thyroid_user",
            password="ar@vv990"
        )
        print("✅ Direct psycopg2 connection: SUCCESS")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Direct psycopg2 connection: FAILED - {e}")
        return False

# Test 2: SQLAlchemy connection
def test_sqlalchemy_connection():
    try:
        DATABASE_URL = "postgresql://thyroid_user:ar%40vv990@localhost/thyroid_aware"
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ SQLAlchemy connection: SUCCESS")
            return True
    except Exception as e:
        print(f"❌ SQLAlchemy connection: FAILED - {e}")
        return False

# Test 3: Check if database and user exist
def test_database_exists():
    try:
        # Connect as postgres user to check if database exists
        conn = psycopg2.connect(
            host="localhost",
            database="postgres",  # Connect to default postgres db
            user="postgres",
            password="your_postgres_password"  # Replace with your postgres password
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'thyroid_aware'")
        db_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT 1 FROM pg_user WHERE usename = 'thyroid_user'")
        user_exists = cursor.fetchone() is not None
        
        print(f"Database 'thyroid_aware' exists: {'✅' if db_exists else '❌'}")
        print(f"User 'thyroid_user' exists: {'✅' if user_exists else '❌'}")
        
        conn.close()
        return db_exists and user_exists
    except Exception as e:
        print(f"❌ Database/User check: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("Testing PostgreSQL Connection...")
    print("=" * 40)
    
    test_psycopg2_connection()
    test_sqlalchemy_connection()
    test_database_exists()
