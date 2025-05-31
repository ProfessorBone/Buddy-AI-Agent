#!/usr/bin/env python3
"""
Trucking Database Handler for TechMentor Agent
Lightning-fast trip logging using basic SQLite3
Based on Walmart Private Fleet Assistant requirements
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re

class TruckingDatabase:
    def __init__(self, db_path: str = "trucking_data.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This makes rows act like dictionaries
        cursor = conn.cursor()
        
        # Create trips table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trips (
            trip_id TEXT PRIMARY KEY,
            trip_number TEXT,
            start_odometer INTEGER,
            end_odometer INTEGER,
            total_miles INTEGER,
            trip_type TEXT,
            start_trailer TEXT,
            end_trailer TEXT,
            origin TEXT,
            num_stops INTEGER,
            start_time TEXT,
            end_time TEXT,
            status TEXT,
            created_at TEXT
        )''')
        
        # Create stops table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stops (
            stop_id TEXT PRIMARY KEY,
            trip_id TEXT,
            stop_number INTEGER,
            location TEXT,
            city TEXT,
            state TEXT,
            break_type TEXT,
            arrival_time TEXT,
            departure_time TEXT,
            created_at TEXT,
            FOREIGN KEY (trip_id) REFERENCES trips (trip_id)
        )''')
        
        # Create events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            trip_id TEXT,
            event_type TEXT,
            event_description TEXT,
            trailer_number TEXT,
            location TEXT,
            timestamp TEXT,
            created_at TEXT,
            FOREIGN KEY (trip_id) REFERENCES trips (trip_id)
        )''')
        
        # Create layovers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS layovers (
            layover_id TEXT PRIMARY KEY,
            trip_id TEXT,
            start_time TEXT,
            end_time TEXT,
            location TEXT,
            duration_hours REAL,
            date TEXT,
            created_at TEXT,
            FOREIGN KEY (trip_id) REFERENCES trips (trip_id)
        )''')
        
        conn.commit()
        conn.close()
        
    def generate_id(self) -> str:
        """Generate unique ID for records"""
        return str(uuid.uuid4())[:8]
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def get_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # ===== TRIP MANAGEMENT =====
    
    def start_trip(self, trip_number: str, start_odometer: int, 
                   trailer: str, origin: str = "") -> Dict:
        """Start a new trip - instant response"""
        trip_id = self.generate_id()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trips (trip_id, trip_number, start_odometer, end_odometer, 
                          total_miles, trip_type, start_trailer, end_trailer, 
                          origin, num_stops, start_time, end_time, status, created_at)
        VALUES (?, ?, ?, 0, 0, '', ?, ?, '', 0, ?, '', 'active', ?)
        ''', (trip_id, trip_number, start_odometer, trailer, trailer, 
              self.get_current_timestamp(), self.get_current_timestamp()))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"✅ Trip {trip_number} started! Odometer: {start_odometer:,}, Trailer: {trailer}",
            "trip_id": trip_id
        }
    
    def end_trip(self, trip_number: str, end_odometer: int) -> Dict:
        """End a trip and calculate miles"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find active trip
        cursor.execute("SELECT * FROM trips WHERE trip_number = ? AND status = 'active'", 
                      (trip_number,))
        trip = cursor.fetchone()
        
        if not trip:
            conn.close()
            return {"success": False, "message": f"❌ No active trip {trip_number} found"}
        
        total_miles = end_odometer - trip["start_odometer"]
        
        # Update trip
        cursor.execute('''
        UPDATE trips SET end_odometer = ?, total_miles = ?, 
                        end_time = ?, status = 'completed' 
        WHERE trip_id = ?
        ''', (end_odometer, total_miles, self.get_current_timestamp(), trip["trip_id"]))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"✅ Trip {trip_number} completed! Miles: {total_miles:,}",
            "trip_id": trip["trip_id"],
            "total_miles": total_miles
        }
    
    def add_stop(self, trip_number: str, location: str, city: str = "", 
                 state: str = "", break_type: str = "No Break") -> Dict:
        """Add stop to trip"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find active trip
        cursor.execute("SELECT * FROM trips WHERE trip_number = ? AND status = 'active'", 
                      (trip_number,))
        trip = cursor.fetchone()
        
        if not trip:
            conn.close()
            return {"success": False, "message": f"❌ No active trip {trip_number}"}
        
        # Get next stop number
        cursor.execute("SELECT COUNT(*) as count FROM stops WHERE trip_id = ?", 
                      (trip["trip_id"],))
        count_result = cursor.fetchone()
        stop_number = count_result["count"] + 1
        
        # Insert stop
        stop_id = self.generate_id()
        cursor.execute('''
        INSERT INTO stops (stop_id, trip_id, stop_number, location, city, state, 
                          break_type, arrival_time, departure_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, '', '', ?)
        ''', (stop_id, trip["trip_id"], stop_number, location, city, state, 
              break_type, self.get_current_timestamp()))
        
        # Update trip stop count
        cursor.execute("UPDATE trips SET num_stops = ? WHERE trip_id = ?", 
                      (stop_number, trip["trip_id"]))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"✅ Added stop {stop_number}: {location} ({break_type})",
            "stop_id": stop_id
        }
    
    def hook_trailer(self, trip_number: str, trailer_number: str, location: str = "") -> Dict:
        """Log trailer hook event"""
        return self.log_event(trip_number, "Hk", f"Hooked to trailer {trailer_number}", 
                             trailer_number, location)
    
    def log_event(self, trip_number: str, event_type: str, 
                  description: str, trailer: str = "", location: str = "") -> Dict:
        """Log trucking events: Hk, D, Ar, Do, LL, LU, LO"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find active trip
        cursor.execute("SELECT * FROM trips WHERE trip_number = ? AND status = 'active'", 
                      (trip_number,))
        trip = cursor.fetchone()
        
        if not trip:
            conn.close()
            return {"success": False, "message": f"❌ No active trip {trip_number}"}
        
        event_id = self.generate_id()
        cursor.execute('''
        INSERT INTO events (event_id, trip_id, event_type, event_description, 
                           trailer_number, location, timestamp, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (event_id, trip["trip_id"], event_type, description, trailer, 
              location, self.get_current_timestamp(), self.get_current_timestamp()))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"✅ {event_type}: {description}",
            "event_id": event_id
        }
    
    # ===== QUERIES & ANALYSIS =====
    
    def get_trip_summary(self, trip_number: str) -> Dict:
        """Get complete trip summary"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trips WHERE trip_number = ?", (trip_number,))
        trip = cursor.fetchone()
        
        if not trip:
            conn.close()
            return {"success": False, "message": f"❌ Trip {trip_number} not found"}
        
        # Get stops
        cursor.execute("SELECT * FROM stops WHERE trip_id = ? ORDER BY stop_number", 
                      (trip["trip_id"],))
        stops = cursor.fetchall()
        
        # Get events
        cursor.execute("SELECT * FROM events WHERE trip_id = ? ORDER BY timestamp", 
                      (trip["trip_id"],))
        events = cursor.fetchall()
        
        conn.close()
        
        return {
            "success": True,
            "trip": dict(trip),
            "stops": [dict(stop) for stop in stops],
            "events": [dict(event) for event in events],
            "summary": f"Trip {trip_number}: {trip['total_miles']} miles, {len(stops)} stops, {len(events)} events"
        }
    
    def get_weekly_summary(self) -> Dict:
        """Get this week's trip summary"""
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trips WHERE created_at >= ? ORDER BY start_time DESC", 
                      (week_ago,))
        trips = cursor.fetchall()
        conn.close()
        
        total_miles = sum(trip["total_miles"] for trip in trips if trip["total_miles"])
        total_trips = len(trips)
        

        return {
            "success": True,
            "message": f"This week: {total_trips} trips, {total_miles:,} miles",
            "period": "Last 7 days", 
            "total_trips": total_trips,
            "total_miles": total_miles,
            "summary": f"This week: {total_trips} trips, {total_miles:,} miles"
        }
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        for table in ["trips", "stops", "events", "layovers"]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            result = cursor.fetchone()
            stats[table] = result["count"]
        
        conn.close()
        
        return {
            "success": True,
            "database_file": self.db_path,
            "tables": stats,
            "total_records": sum(stats.values())
        }

# ===== COMMAND PARSER =====

class TruckingCommandParser:
    def __init__(self, db: TruckingDatabase):
        self.db = db
    
    def parse_command(self, text: str) -> Dict:
        """Parse natural language trucking commands"""
        text = text.lower().strip()
        
        # Start trip: "start trip 48220. odometer 702500. trailer 830112"
        start_match = re.search(r'start trip (\d+).*odometer.*?(\d+).*trailer.*?(\w+)', text)
        if start_match:
            trip_num, odometer, trailer = start_match.groups()
            return self.db.start_trip(trip_num, int(odometer), trailer)
        
        # End trip: "trip 48220 ended. odometer 703250"
        end_match = re.search(r'trip (\d+) ended.*odometer.*?(\d+)', text)
        if end_match:
            trip_num, odometer = end_match.groups()
            return self.db.end_trip(trip_num, int(odometer))
        
        # Add stop: "add stop store 2210 dallas okay break"
        stop_match = re.search(r'add stop ([\w\s]+?)(?:\s+(okay break|no break))?$', text)
        if stop_match:
            location = stop_match.group(1).strip()
            break_type = stop_match.group(2) or "No Break"
            # Need trip context - simplified for now
            return {"success": False, "message": "Need trip number to add stop"}
        
        # Weekly summary
        if 'summary' in text and ('week' in text or 'weekly' in text):
            return self.db.get_weekly_summary()
        
        # Trip summary
        summary_match = re.search(r'summary.*trip (\d+)', text)
        if summary_match:
            trip_num = summary_match.group(1)
            return self.db.get_trip_summary(trip_num)
        
        # Database stats
        if 'database' in text and ('stats' in text or 'status' in text):
            return self.db.get_database_stats()
        
        return {
            "success": False, 
            "message": "Command not recognized. Try: 'Start trip 48220. Odometer 702500. Trailer 830112'"
        }

# ===== TESTING FUNCTION =====

def test_database():
    """Test the trucking database with sample data"""
    print("🧪 Testing Trucking Database:")
    
    db = TruckingDatabase()
    
    print("\n1. Starting trip...")
    result = db.start_trip("48220", 702500, "830112", "DC 7026")
    print(f"   {result['message']}")
    
    print("\n2. Adding stop...")
    result = db.add_stop("48220", "Store 2210", "Dallas", "TX", "Okay Break")
    print(f"   {result['message']}")
    
    print("\n3. Logging hook event...")
    result = db.hook_trailer("48220", "904123", "Store 2210")
    print(f"   {result['message']}")
    
    print("\n4. Ending trip...")
    result = db.end_trip("48220", 703250)
    print(f"   {result['message']}")
    
    print("\n5. Trip summary...")
    result = db.get_trip_summary("48220")
    print(f"   {result['summary']}")
    
    print("\n6. Weekly summary...")
    result = db.get_weekly_summary()
    print(f"   {result['summary']}")
    
    print("\n7. Database stats...")
    stats = db.get_database_stats()
    print(f"   Database: {stats['total_records']} total records")
    print(f"   Tables: {stats['tables']}")
    
    print("\n✅ Database testing complete!")
    return db

if __name__ == "__main__":
    test_database()
