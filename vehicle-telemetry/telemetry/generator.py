"""
Simulation von Fahrzeug-Telemetrie (Sensordaten).

Echte Fahrzeugdaten sind nicht frei verfügbar. Deshalb werden hier Messwerte
simuliert, die sich ähnlich verhalten wie die Signale eines echten Autos:

    - Geschwindigkeit und Drehzahl (hängen zusammen)
    - Kühlmitteltemperatur, Batteriespannung, Reifendruck

In die normalen Daten bauen wir gezielt Fehler ein ("Fault Injection").
Dadurch wissen wir später genau, wo die Anomalien wirklich liegen, und können
messen, wie gut die Erkennung funktioniert.
"""

import numpy as np
import pandas as pd

# --- Einstellungen der Simulation -------------------------------------------
SAMPLE_INTERVAL_S = 5          # alle 5 Sekunden ein Messwert
N_SAMPLES = 1440               # 1440 * 5 s = 2 Stunden Fahrt
START_TIME = "2026-01-15 08:00:00"

# Reihenfolge der Spalten (wird auch von der Datenbank verwendet)
COLUMNS = [
    "vehicle_id",
    "timestamp",
    "speed_kmh",
    "rpm",
    "coolant_temp_c",
    "battery_voltage_v",
    "tire_pressure_bar",
    "fault_type",
]

# Welche Fehler gibt es? Jeder Fehler verändert genau ein Signal:
#   Fehlername -> (betroffenes Signal, maximale Abweichung vom Normalwert)
FAULT_EFFECTS = {
    "overheating": ("coolant_temp_c", +25.0),    # Motor wird zu heiß
    "low_battery": ("battery_voltage_v", -2.0),  # Bordnetzspannung bricht ein
    "tire_leak": ("tire_pressure_bar", -0.8),    # Reifen verliert Luft
}

# Unsere kleine Flotte: 3 Fahrzeuge mit je 1-2 eingebauten Fehlern.
# Ein Fehler ist ein Tupel: (Fehlerart, Startindex, Länge in Messwerten)
FLEET = [
    {"vehicle_id": 1, "name": "Fahrzeug A",
     "faults": [("overheating", 500, 100)]},
    {"vehicle_id": 2, "name": "Fahrzeug B",
     "faults": [("low_battery", 300, 80), ("tire_leak", 900, 100)]},
    {"vehicle_id": 3, "name": "Fahrzeug C",
     "faults": [("overheating", 250, 80), ("tire_leak", 1000, 100)]},
]


def simulate_vehicle(vehicle_id, seed, faults, n_samples=N_SAMPLES):
    """Simuliert eine Fahrt eines Fahrzeugs und gibt einen DataFrame zurück.

    vehicle_id : Nummer des Fahrzeugs
    seed       : Startwert des Zufallsgenerators -> gleicher Seed = gleiche Daten
    faults     : Liste von (Fehlerart, Startindex, Länge), siehe FLEET
    n_samples  : Anzahl der Messwerte
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)  # 0, 1, 2, ... (Zeitindex)

    # 1) Geschwindigkeit: Mischung aus zwei langsamen Wellen (Landstraße/Stadt)
    #    plus Rauschen. Zusätzlich ein paar "Ampelstopps" mit 0 km/h.
    speed = (
        70
        + 35 * np.sin(2 * np.pi * t / 300 + rng.uniform(0, 2 * np.pi))
        + 20 * np.sin(2 * np.pi * t / 90 + rng.uniform(0, 2 * np.pi))
        + rng.normal(0, 3, n_samples)
    )
    for start in rng.integers(0, n_samples - 10, size=6):
        speed[start:start + 8] = 0
    speed = np.clip(speed, 0, 160)

    # 2) Drehzahl: Leerlauf (800 U/min) + Anteil, der mit der Geschwindigkeit wächst
    rpm = np.clip(800 + 22 * speed + rng.normal(0, 60, n_samples), 700, None)

    # 3) Gesundheits-Signale im Normalbetrieb
    signals = {
        # Motor ist fast warm und pendelt sich bei ca. 90 °C ein;
        # bei höherer Geschwindigkeit wird er minimal wärmer.
        "coolant_temp_c": 90 - 6 * np.exp(-t / 60) + 0.04 * (speed - 70)
                          + rng.normal(0, 0.8, n_samples),
        # Lichtmaschine hält die Bordspannung bei ca. 13,8 V
        "battery_voltage_v": 13.8 + rng.normal(0, 0.1, n_samples),
        # Reifendruck ca. 2,4 bar mit minimalem Messrauschen
        "tire_pressure_bar": 2.4 + rng.normal(0, 0.02, n_samples),
    }

    # 4) Fehler einbauen ("Fault Injection") und in 'fault_type' markieren
    fault_type = np.full(n_samples, "none", dtype=object)
    for kind, start, length in faults:
        column, max_change = FAULT_EFFECTS[kind]
        end = min(start + length, n_samples)
        window = end - start
        # Form des Fehlers: steigt in der ersten Hälfte an, bleibt dann konstant
        shape = np.minimum(np.linspace(0, 1, window) * 2, 1)
        signals[column][start:end] += max_change * shape
        fault_type[start:end] = kind

    # 5) Alles in eine Tabelle packen
    df = pd.DataFrame({
        "vehicle_id": vehicle_id,
        "timestamp": pd.date_range(START_TIME, periods=n_samples,
                                   freq=f"{SAMPLE_INTERVAL_S}s"),
        "speed_kmh": speed.round(1),
        "rpm": rpm.round(0),
        "coolant_temp_c": signals["coolant_temp_c"].round(2),
        "battery_voltage_v": signals["battery_voltage_v"].round(2),
        "tire_pressure_bar": signals["tire_pressure_bar"].round(3),
        "fault_type": fault_type,
    })
    return df[COLUMNS]


def simulate_fleet(seed=42):
    """Simuliert alle Fahrzeuge aus FLEET.

    Gibt zurück: (Liste von (vehicle_id, name), DataFrame mit allen Messwerten)
    """
    vehicles = [(v["vehicle_id"], v["name"]) for v in FLEET]
    frames = [
        # Jedes Fahrzeug bekommt einen eigenen Seed, sonst wären alle identisch
        simulate_vehicle(v["vehicle_id"], seed + v["vehicle_id"], v["faults"])
        for v in FLEET
    ]
    return vehicles, pd.concat(frames, ignore_index=True)
