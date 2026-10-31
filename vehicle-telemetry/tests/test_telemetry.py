"""
Automatische Tests. Ausführen mit:   pytest

Jeder Test prüft eine kleine, klar abgegrenzte Aussage über den Code.
So fällt sofort auf, wenn eine spätere Änderung etwas kaputt macht.
"""

import numpy as np
import pandas as pd

from telemetry import anomaly, database, generator


# --- Simulation --------------------------------------------------------------

def test_simulation_has_expected_shape():
    df = generator.simulate_vehicle(vehicle_id=1, seed=1, faults=[], n_samples=200)
    assert len(df) == 200
    assert list(df.columns) == generator.COLUMNS
    assert df.isna().sum().sum() == 0           # keine fehlenden Werte
    assert (df["speed_kmh"] >= 0).all()         # keine negative Geschwindigkeit


def test_same_seed_gives_same_data():
    a = generator.simulate_vehicle(1, seed=7, faults=[], n_samples=100)
    b = generator.simulate_vehicle(1, seed=7, faults=[], n_samples=100)
    c = generator.simulate_vehicle(1, seed=8, faults=[], n_samples=100)
    assert a.equals(b)        # gleicher Seed -> identische Daten
    assert not a.equals(c)    # anderer Seed -> andere Daten


def test_injected_fault_is_labeled_and_visible():
    faults = [("overheating", 100, 60)]
    df = generator.simulate_vehicle(1, seed=3, faults=faults, n_samples=300)

    in_window = df["fault_type"] == "overheating"
    assert in_window.sum() == 60                          # genau 60 Messwerte markiert
    assert df.loc[in_window, "coolant_temp_c"].max() > 105    # Fehler ist deutlich sichtbar
    assert df.loc[~in_window, "coolant_temp_c"].max() < 100   # sonst ist alles normal


# --- Datenbank ---------------------------------------------------------------

def test_database_roundtrip():
    vehicles, measurements = generator.simulate_fleet(seed=42)
    conn = database.connect(":memory:")   # Datenbank nur im Arbeitsspeicher
    database.save_fleet(conn, vehicles, measurements)

    assert database.count_measurements(conn) == len(measurements)
    assert len(database.list_vehicles(conn)) == 3

    loaded = database.load_measurements(conn, vehicle_id=1)
    assert len(loaded) == generator.N_SAMPLES
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    conn.close()


def test_summary_has_one_row_per_vehicle():
    vehicles, measurements = generator.simulate_fleet(seed=42)
    conn = database.connect(":memory:")
    database.save_fleet(conn, vehicles, measurements)

    summary = database.summary_per_vehicle(conn)
    assert len(summary) == 3
    assert (summary["Messwerte"] == generator.N_SAMPLES).all()
    conn.close()


def test_reset_empties_database():
    vehicles, measurements = generator.simulate_fleet(seed=42)
    conn = database.connect(":memory:")
    database.save_fleet(conn, vehicles, measurements)
    database.reset(conn)
    assert database.count_measurements(conn) == 0
    conn.close()


# --- Anomalie-Erkennung ------------------------------------------------------

def test_rules_raise_no_alarm_on_healthy_data():
    df = generator.simulate_vehicle(1, seed=5, faults=[])
    assert anomaly.detect_with_rules(df).sum() == 0


def test_rules_find_overheating():
    faults = [("overheating", 400, 100)]
    df = generator.simulate_vehicle(1, seed=5, faults=faults)
    truth = (df["fault_type"] != "none").to_numpy()

    result = anomaly.evaluate(truth, anomaly.detect_with_rules(df))
    assert result["precision"] == 1.0     # kein Fehlalarm
    assert result["recall"] > 0.5         # Großteil des Fehlers gefunden


def test_isolation_forest_flags_expected_share():
    df = generator.simulate_vehicle(1, seed=5, faults=[("low_battery", 300, 80)])
    flags = anomaly.detect_with_isolation_forest(df, contamination=0.05)

    assert flags.dtype == bool
    assert len(flags) == len(df)
    assert abs(flags.mean() - 0.05) < 0.01    # ca. 5 % der Messwerte markiert


def test_evaluate_computes_metrics_correctly():
    truth = np.array([True, True, False, False])
    flags = np.array([True, False, True, False])
    result = anomaly.evaluate(truth, flags)
    # 1 Treffer, 1 Fehlalarm, 1 übersehen -> alles 50 %
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_evaluate_handles_no_alarms_without_crash():
    result = anomaly.evaluate([True, False], [False, False])
    assert result["precision"] == 0.0 and result["recall"] == 0.0


def test_find_windows_groups_consecutive_flags():
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=5, freq="5s"))
    windows = anomaly.find_windows([False, True, True, False, True], timestamps)
    assert windows == [(timestamps[1], timestamps[2]), (timestamps[4], timestamps[4])]
