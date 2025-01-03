import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Load environment variables
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT'),
    'database': os.getenv('POSTGRES_DB')
}

# Create database if it doesn't exist
def create_database():
    # Connect to default postgres database
    conn = psycopg2.connect(
        user=DB_PARAMS['user'],
        password=DB_PARAMS['password'],
        host=DB_PARAMS['host'],
        port=DB_PARAMS['port'],
        database='postgres'
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_PARAMS['database']}'")
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(f"CREATE DATABASE {DB_PARAMS['database']}")
    
    cursor.close()
    conn.close()

# SQLAlchemy setup
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def init_db():
    # Create database if it doesn't exist
    create_database()
    
    # Create SQLAlchemy engine
    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@"
        f"{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
    )
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    
    return Session()

if __name__ == "__main__":
    session = init_db()
    print("Database initialized successfully!")
