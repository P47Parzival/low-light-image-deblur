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
            total_wagons INTEGER DEFAULT 0,
            enhanced_video_path TEXT,
            status TEXT DEFAULT 'PROCESSING'
        )
    ''')
    
    # Try to add columns if they don't exist (Migration)
    try:
        cursor.execute('ALTER TABLE inspections ADD COLUMN enhanced_video_path TEXT')
    except sqlite3.OperationalError:
        pass 
    
    try:
        cursor.execute("ALTER TABLE inspections ADD COLUMN status TEXT DEFAULT 'PROCESSING'")
    except sqlite3.OperationalError:
        pass

    # New Metrics Migrations
    try:
        cursor.execute("ALTER TABLE inspections ADD COLUMN fps REAL")
        cursor.execute("ALTER TABLE inspections ADD COLUMN resolution TEXT")
        cursor.execute("ALTER TABLE inspections ADD COLUMN avg_brightness REAL")
        cursor.execute("ALTER TABLE inspections ADD COLUMN blur_stats TEXT") # JSON string
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE inspections ADD COLUMN train_speed REAL")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE wagons ADD COLUMN anomaly_image_path TEXT")
        cursor.execute("ALTER TABLE wagons ADD COLUMN anomaly_type TEXT")
        cursor.execute("ALTER TABLE wagons ADD COLUMN anomaly_confidence REAL")
    except sqlite3.OperationalError:
        pass
    
    try:
        cursor.execute("ALTER TABLE inspections ADD COLUMN start_time TEXT")
        cursor.execute("ALTER TABLE inspections ADD COLUMN end_time TEXT")
    except sqlite3.OperationalError:
        pass
    
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
            anomaly_image_path TEXT,
            anomaly_type TEXT,
            anomaly_confidence REAL,
            FOREIGN KEY (inspection_id) REFERENCES inspections (id)
        )
    ''')
    
    conn.commit()
    return conn

def update_inspection_times(inspection_id, start_time=None, end_time=None):
    """Update start and end times for an inspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if start_time:
        cursor.execute('UPDATE inspections SET start_time = ? WHERE id = ?', (start_time, inspection_id))
    if end_time:
        cursor.execute('UPDATE inspections SET end_time = ? WHERE id = ?', (end_time, inspection_id))
        
    conn.commit()
    conn.close()

def update_inspection_video_path(inspection_id, video_path):
    """Update the enhanced video path for an inspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE inspections SET enhanced_video_path = ? WHERE id = ?', (video_path, inspection_id))
    conn.commit()
    conn.close()

def update_inspection_status(inspection_id, status):
    """Update the status of an inspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE inspections SET status = ? WHERE id = ?', (status, inspection_id))
    conn.commit()
    conn.close()

def update_inspection_metrics(inspection_id, fps, resolution, brightness, blur_stats, train_speed=0.0):
    """Update the system health metrics for an inspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE inspections 
        SET fps = ?, resolution = ?, avg_brightness = ?, blur_stats = ?, train_speed = ? 
        WHERE id = ?
    ''', (fps, resolution, brightness, blur_stats, train_speed, inspection_id))
    conn.commit()
    conn.close()

def create_inspection(video_name):
    """Create a new inspection record and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO inspections (video_name, timestamp, status) VALUES (?, ?, ?)', (video_name, timestamp, 'PROCESSING'))
    
    conn.commit()
    inspection_id = cursor.lastrowid
    conn.close()
    return inspection_id

def update_inspection_count(inspection_id, total_wagons):
    """Update the total wagon count for an inspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('UPDATE inspections SET total_wagons = ? WHERE id = ?', (total_wagons, inspection_id))
    
    conn.commit()
    conn.close()

def add_wagon(inspection_id, wagon_index, ocr_text, ocr_conf, orig_path, deblur_path, ocr_path, defects, is_night, anomaly_path="", anomaly_type="", anomaly_conf=0.0):
    """Add a wagon record to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO wagons 
        (inspection_id, wagon_index, ocr_text, ocr_confidence, original_image_path, deblurred_image_path, cropped_number_path, defects, is_night, timestamp, anomaly_image_path, anomaly_type, anomaly_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (inspection_id, wagon_index, ocr_text, ocr_conf, orig_path, deblur_path, ocr_path, str(defects), is_night, timestamp, anomaly_path, anomaly_type, anomaly_conf))
    
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
