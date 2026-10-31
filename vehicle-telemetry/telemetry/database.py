"""
Speichern und Abfragen der Messwerte in einer SQLite-Datenbank.

SQLite ist eine Datenbank in einer einzigen Datei (data/telemetry.db) und
braucht keinen Server. Für dieses Projekt ist das ideal; die SQL-Befehle wären
aber fast identisch bei PostgreSQL oder MySQL.

Tabellen:
    vehicles      (1 Zeile pro Fahrzeug)
    measurements  (1 Zeile pro Messung, verweist per Fremdschlüssel auf vehicles)
"""

import sqlite3
from pathlib import Path

import pandas as pd

from telemetry import generator

DB_PATH = Path("data") / "telemetry.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id        INTEGER NOT NULL,
    timestamp         TEXT    NOT NULL,
    speed_kmh         REAL    NOT NULL,
    rpm               REAL    NOT NULL,
    coolant_temp_c    REAL    NOT NULL,
    battery_voltage_v REAL    NOT NULL,
    tire_pressure_bar REAL    NOT NULL,
    fault_type        TEXT    NOT NULL DEFAULT 'none',
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id)
);

-- Index: macht Abfragen "alle Messwerte von Fahrzeug X" schnell
CREATE INDEX IF NOT EXISTS idx_measurements_vehicle_time
    ON measurements (vehicle_id, timestamp);
"""

# Abfrage 1: alle Messwerte eines Fahrzeugs in zeitlicher Reihenfolge
QUERY_LOAD = """
SELECT timestamp, speed_kmh, rpm, coolant_temp_c,
       battery_voltage_v, tire_pressure_bar, fault_type
FROM   measurements
WHERE  vehicle_id = ?
ORDER  BY timestamp
"""

# Abfrage 2: Übersicht pro Fahrzeug (JOIN + GROUP BY + Aggregatfunktionen)
QUERY_SUMMARY = """
SELECT v.name                           AS Fahrzeug,
       COUNT(*)                         AS Messwerte,
       ROUND(AVG(m.coolant_temp_c), 1)  AS "Ø Temperatur (°C)",
       MAX(m.coolant_temp_c)            AS "Max. Temperatur (°C)",
       MIN(m.battery_voltage_v)         AS "Min. Spannung (V)",
       MIN(m.tire_pressure_bar)         AS "Min. Reifendruck (bar)"
FROM   measurements m
JOIN   vehicles v ON v.vehicle_id = m.vehicle_id
GROUP  BY v.vehicle_id
ORDER  BY v.vehicle_id
"""


def connect(db_path=DB_PATH):
    """Öffnet die Datenbank (und legt Datei und Tabellen an, falls nötig)."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # Fremdschlüssel wirklich prüfen
    conn.executescript(SCHEMA)
    return conn


def reset(conn):
    """Löscht alle Daten (Tabellen bleiben bestehen)."""
    conn.execute("DELETE FROM measurements")
    conn.execute("DELETE FROM vehicles")
    conn.commit()


def save_fleet(conn, vehicles, measurements):
    """Speichert Fahrzeuge und Messwerte.

    vehicles     : Liste von (vehicle_id, name)
    measurements : DataFrame mit den Spalten aus generator.COLUMNS
    """
    conn.executemany(
        "INSERT INTO vehicles (vehicle_id, name) VALUES (?, ?)", vehicles
    )

    rows = measurements[generator.COLUMNS].copy()
    # SQLite kennt keinen Datums-Typ -> wir speichern Text im Format JJJJ-MM-TT hh:mm:ss
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    conn.executemany(
        """
        INSERT INTO measurements
            (vehicle_id, timestamp, speed_kmh, rpm, coolant_temp_c,
             battery_voltage_v, tire_pressure_bar, fault_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows.astype(object).values.tolist(),  # normale Python-Werte für sqlite3
    )
    conn.commit()


def count_measurements(conn):
    """Anzahl aller gespeicherten Messwerte."""
    return conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]


def list_vehicles(conn):
    """Alle Fahrzeuge als DataFrame (vehicle_id, name)."""
    return pd.read_sql_query(
        "SELECT vehicle_id, name FROM vehicles ORDER BY vehicle_id", conn
    )


def load_measurements(conn, vehicle_id):
    """Lädt alle Messwerte eines Fahrzeugs als DataFrame."""
    # Das "?" ist ein Platzhalter. So werden Werte sicher eingesetzt
    # (Schutz vor SQL-Injection) - nie Werte per f-String in SQL einbauen!
    df = pd.read_sql_query(QUERY_LOAD, conn, params=(int(vehicle_id),))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def summary_per_vehicle(conn):
    """Kennzahlen pro Fahrzeug (siehe QUERY_SUMMARY)."""
    return pd.read_sql_query(QUERY_SUMMARY, conn)
