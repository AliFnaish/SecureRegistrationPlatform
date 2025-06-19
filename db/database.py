import sqlite3
from datetime import datetime

DB_PATH = "db/registration.db"

def create_connection():
    return sqlite3.connect(DB_PATH)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            father_name TEXT,
            family_name TEXT,
            mother_name TEXT,
            place_birth TEXT,
            image_path TEXT,
            embedding TEXT,
            created_at TEXT
        )
    ''')

    conn.commit()
    conn.close()

def insert_user(name, father_name, family_name, mother_name, place_birth, image_path, embedding):
    conn = create_connection()
    cursor = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO users (
            name, father_name, family_name, mother_name,
            place_birth, image_path, embedding, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, father_name, family_name, mother_name, place_birth, image_path, embedding, created_at))

    conn.commit()
    conn.close()

def get_all_users():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    rows = cursor.fetchall()
    conn.close()
    return rows
