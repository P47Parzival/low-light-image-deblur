import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '../../full model/detection/inspections.db')

def init_db():
    """Initialize the database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table: Inspections (Represents a single video run)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_wagons INTEGER DEFAULT 0
        )
    ''')
    
    # Table: Wagons (Represents a detected wagon in an inspection)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wagons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id INTEGER NOT NULL,
            wagon_index INTEGER NOT NULL,
            ocr_text TEXT,
            ocr_confidence REAL,
            original_image_path TEXT,
            deblurred_image_path TEXT,
            cropped_number_path TEXT,
            defects TEXT,
            is_night BOOLEAN DEFAULT 0,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (inspection_id) REFERENCES inspections (id)
        )
    ''')
    
    conn.commit()
    return conn

def create_inspection(video_name):
    """Create a new inspection record and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO inspections (video_name, timestamp) VALUES (?, ?)', (video_name, timestamp))
    
    conn.commit()
    inspection_id = cursor.lastrowid
    conn.close()
    return inspection_id

def add_wagon(inspection_id, wagon_index, ocr_text, ocr_conf, orig_path, deblur_path, ocr_path, defects, is_night):
    """Add a wagon record to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO wagons 
        (inspection_id, wagon_index, ocr_text, ocr_confidence, original_image_path, deblurred_image_path, cropped_number_path, defects, is_night, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (inspection_id, wagon_index, ocr_text, ocr_conf, orig_path, deblur_path, ocr_path, str(defects), is_night, timestamp))
    
    conn.commit()
    conn.close()

def get_all_inspections():
    """Fetch all inspections ordered by date."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM inspections ORDER BY id DESC')
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_wagons_for_inspection(inspection_id):
    """Fetch all wagons for a specific inspection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM wagons WHERE inspection_id = ? ORDER BY wagon_index ASC', (inspection_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_inspection_by_id(inspection_id):
    """Fetch a single inspection by its ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM inspections WHERE id = ?', (inspection_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
