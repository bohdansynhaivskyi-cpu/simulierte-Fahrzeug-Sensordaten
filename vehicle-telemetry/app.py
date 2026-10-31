"""
Dashboard für die Fahrzeug-Telemetrie.

Start:   streamlit run app.py

Ablauf bei jedem Klick im Dashboard (Streamlit führt dieses Skript neu aus):
    1. Beim allerersten Start: Daten simulieren und in SQLite speichern
    2. Messwerte des gewählten Fahrzeugs per SQL aus der Datenbank laden
    3. Anomalien erkennen (Isolation Forest oder Grenzwert-Regeln)
    4. Ergebnis als Kennzahlen und Diagramm anzeigen
"""

from contextlib import closing

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from telemetry import anomaly, database, generator

st.set_page_config(page_title="Fahrzeug-Telemetrie", page_icon="🚗", layout="wide")

# Diese Signale werden gezeichnet: (Spaltenname, Beschriftung)
SIGNALS = [
    ("coolant_temp_c", "Kühlmitteltemperatur (°C)"),
    ("battery_voltage_v", "Batteriespannung (V)"),
    ("tire_pressure_bar", "Reifendruck (bar)"),
    ("speed_kmh", "Geschwindigkeit (km/h)"),
    ("rpm", "Drehzahl (U/min)"),
]

METHOD_ML = "Isolation Forest (Machine Learning)"
METHOD_RULES = "Grenzwert-Regeln"


def regenerate(seed):
    """Simuliert neue Daten und ersetzt damit den Inhalt der Datenbank."""
    vehicles, measurements = generator.simulate_fleet(seed)
    with closing(database.connect()) as conn:
        database.reset(conn)
        database.save_fleet(conn, vehicles, measurements)


def build_figure(df, flags, truth, show_truth):
    """Baut das Diagramm: eine Zeile pro Signal, rote Punkte = erkannte Anomalien."""
    fig = make_subplots(
        rows=len(SIGNALS), cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[label for _, label in SIGNALS],
    )

    for row, (column, label) in enumerate(SIGNALS, start=1):
        # Linie: der gemessene Verlauf
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[column], mode="lines",
                       line=dict(width=1, color="#4c78a8"), showlegend=False),
            row=row, col=1,
        )
        # Rote Punkte: Messwerte, die als Anomalie erkannt wurden
        fig.add_trace(
            go.Scatter(x=df.loc[flags, "timestamp"], y=df.loc[flags, column],
                       mode="markers", marker=dict(color="red", size=5),
                       name="erkannte Anomalie", showlegend=(row == 1)),
            row=row, col=1,
        )

    # Orange Hintergrund: Zeitfenster, in denen die Simulation wirklich einen Fehler hat
    if show_truth:
        for start, end in anomaly.find_windows(truth, df["timestamp"]):
            for row in range(1, len(SIGNALS) + 1):
                fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.2,
                              line_width=0, row=row, col=1)

    fig.update_layout(height=850, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.04))
    return fig


# --- Kopf der Seite ----------------------------------------------------------
st.title("🚗 Fahrzeug-Telemetrie: Anomalie-Erkennung")
st.caption(
    "Simulierte Sensordaten von drei Fahrzeugen, gespeichert in SQLite. "
    "Ziel: Fehler wie Überhitzung, schwache Batterie oder Reifenverlust "
    "automatisch erkennen (Predictive Maintenance)."
)

# --- Seitenleiste: Daten -----------------------------------------------------
st.sidebar.header("1. Daten")
seed = st.sidebar.number_input(
    "Seed der Simulation", min_value=0, value=42, step=1,
    help="Gleicher Seed = gleiche Daten. Erst mit dem Button unten werden neue Daten erzeugt.",
)
if st.sidebar.button("Daten neu simulieren"):
    regenerate(int(seed))

# Beim allerersten Start ist die Datenbank noch leer -> einmal simulieren
with closing(database.connect()) as conn:
    is_empty = database.count_measurements(conn) == 0
if is_empty:
    regenerate(int(seed))

with closing(database.connect()) as conn:
    vehicles = database.list_vehicles(conn)
    summary = database.summary_per_vehicle(conn)

names = dict(zip(vehicles["vehicle_id"].tolist(), vehicles["name"].tolist()))

# --- Seitenleiste: Auswertung ------------------------------------------------
st.sidebar.header("2. Auswertung")
vehicle_id = st.sidebar.selectbox(
    "Fahrzeug", options=list(names), format_func=lambda i: names[i]
)
method = st.sidebar.radio("Methode", [METHOD_ML, METHOD_RULES])
contamination = 0.08
if method == METHOD_ML:
    contamination = st.sidebar.slider(
        "Erwarteter Anteil an Anomalien", min_value=0.01, max_value=0.20,
        value=0.08, step=0.01,
        help="Der Isolation Forest markiert genau diesen Anteil der auffälligsten Messwerte.",
    )
show_truth = st.sidebar.checkbox("Eingebaute Fehler anzeigen (orange)", value=True)

# --- Daten laden und auswerten -----------------------------------------------
with closing(database.connect()) as conn:
    df = database.load_measurements(conn, vehicle_id)

if method == METHOD_ML:
    flags = anomaly.detect_with_isolation_forest(df, contamination=contamination)
else:
    flags = anomaly.detect_with_rules(df)

truth = (df["fault_type"] != "none").to_numpy()   # Wahrheit aus der Simulation
metrics = anomaly.evaluate(truth, flags)

# --- Kennzahlen ----------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Messwerte", f"{len(df):,}".replace(",", "."))
col2.metric("Erkannte Anomalien", int(flags.sum()))
col3.metric("Precision", f"{metrics['precision']:.0%}",
            help="Wie viele der Alarme waren berechtigt? (wenig Fehlalarme = hoch)")
col4.metric("Recall", f"{metrics['recall']:.0%}",
            help="Wie viele der echten Fehler wurden gefunden? (wenig übersehen = hoch)")

# --- Diagramm ------------------------------------------------------------------
st.plotly_chart(build_figure(df, flags, truth, show_truth))
st.caption(
    "Rote Punkte = erkannte Anomalien. Oranger Hintergrund = Zeitfenster, in dem die "
    "Simulation tatsächlich einen Fehler eingebaut hat. Am Anfang eines Fehlers ist die "
    "Abweichung noch klein und deshalb schwer zu erkennen. Ein roter Punkt gilt für den "
    "ganzen Messzeitpunkt und erscheint deshalb in allen Diagrammen, auch wenn nur ein "
    "Signal auffällig ist."
)

# --- Zusatzinfos zum Aufklappen ----------------------------------------------
st.subheader("Übersicht aus der Datenbank")
st.dataframe(summary, hide_index=True)

with st.expander("Die SQL-Abfragen hinter dieser Seite"):
    st.markdown("**Messwerte eines Fahrzeugs laden**")
    st.code(database.QUERY_LOAD, language="sql")
    st.markdown("**Übersicht pro Fahrzeug (JOIN + GROUP BY)**")
    st.code(database.QUERY_SUMMARY, language="sql")

with st.expander("Wie funktioniert die Erkennung?"):
    st.markdown(
        """
**Grenzwert-Regeln:** feste Schwellen, z. B. Kühlmitteltemperatur über 105 °C.
Einfach und gut erklärbar, aber die Schwellen müssen von Hand festgelegt werden.

**Isolation Forest:** Der Algorithmus teilt die Daten mit vielen zufälligen Schnitten.
Auffällige Messwerte liegen allein am Rand und sind schon nach wenigen Schnitten
isoliert. Es sind keine festen Grenzwerte nötig.

**Precision und Recall:** Precision = Anteil berechtigter Alarme. Recall = Anteil der
gefundenen Fehler. Beide zusammen sagen, wie gut eine Methode wirklich ist.
        """
    )

with st.expander("Rohdaten ansehen"):
    st.dataframe(df, hide_index=True)
