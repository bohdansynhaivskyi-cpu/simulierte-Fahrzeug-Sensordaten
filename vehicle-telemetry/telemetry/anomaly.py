"""
Erkennung von Anomalien (= auffällige Messwerte).

Zwei Methoden zum Vergleich:

1) Grenzwert-Regeln   : einfach, leicht zu erklären ("Temperatur > 105 °C").
2) Isolation Forest   : Machine-Learning-Verfahren, das ohne feste Grenzwerte
                        auskommt und "ungewöhnliche" Kombinationen findet.

Beide Funktionen geben eine Liste aus True/False zurück (eine Angabe pro
Messwert): True = "Anomalie".
"""

import numpy as np
from sklearn.ensemble import IsolationForest

# Diese Signale beschreiben den "Gesundheitszustand" des Fahrzeugs.
# Geschwindigkeit und Drehzahl sind nur Kontext und kein Fehlerindikator.
FEATURES = ["coolant_temp_c", "battery_voltage_v", "tire_pressure_bar"]

# Feste Grenzwerte: Signal -> (Minimum, Maximum). None = keine Grenze.
THRESHOLDS = {
    "coolant_temp_c": (None, 105.0),
    "battery_voltage_v": (12.5, None),
    "tire_pressure_bar": (2.0, None),
}


def detect_with_rules(df):
    """Markiert Messwerte, die einen festen Grenzwert verletzen."""
    flags = np.zeros(len(df), dtype=bool)
    for column, (minimum, maximum) in THRESHOLDS.items():
        if minimum is not None:
            flags |= (df[column] < minimum).to_numpy()
        if maximum is not None:
            flags |= (df[column] > maximum).to_numpy()
    return flags


def detect_with_isolation_forest(df, contamination=0.08, random_state=42):
    """Markiert Ausreißer mit einem Isolation Forest.

    Idee: Der Algorithmus teilt die Daten mit vielen zufälligen Schnitten
    ("Bäume"). Ungewöhnliche Punkte liegen allein und weit weg vom Rest und
    sind deshalb nach wenigen Schnitten isoliert. Je schneller ein Punkt
    isoliert ist, desto auffälliger ist er.

    contamination : geschätzter Anteil der Anomalien (0.08 = 8 %). Der
                    Algorithmus markiert genau diesen Anteil der auffälligsten
                    Messwerte.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,  # gleiche Ergebnisse bei jedem Lauf
    )
    labels = model.fit_predict(df[FEATURES])  # -1 = Anomalie, 1 = normal
    return labels == -1


def evaluate(truth, flags):
    """Vergleicht die Erkennung mit der Wahrheit aus der Simulation.

    truth : True, wo wirklich ein Fehler eingebaut wurde
    flags : True, wo die Methode eine Anomalie meldet

    Precision: Wie viele der Alarme waren berechtigt?   TP / (TP + FP)
    Recall   : Wie viele echte Fehler wurden gefunden?  TP / (TP + FN)
    F1       : Mittelwert aus beiden (hoch nur, wenn beide hoch sind)
    """
    truth = np.asarray(truth, dtype=bool)
    flags = np.asarray(flags, dtype=bool)

    tp = int(np.sum(truth & flags))    # echter Fehler, erkannt
    fp = int(np.sum(~truth & flags))   # Fehlalarm
    fn = int(np.sum(truth & ~flags))   # echter Fehler, übersehen

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def find_windows(mask, timestamps):
    """Fasst aufeinanderfolgende True-Werte zu Zeitfenstern zusammen.

    Beispiel: [F, T, T, F, T] -> [(Zeit 1, Zeit 2), (Zeit 4, Zeit 4)]
    """
    windows = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            windows.append((timestamps.iloc[start], timestamps.iloc[i - 1]))
            start = None
    if start is not None:
        windows.append((timestamps.iloc[start], timestamps.iloc[len(mask) - 1]))
    return windows
