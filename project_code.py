import pandas as pd
import numpy as np
import time
import random
import webbrowser
from datetime import datetime
from pymongo import MongoClient, errors
from colorama import Fore, Style, init

# ─────────────────────────────────────────────────────────────────
# SECTION 1: Initialize
# ─────────────────────────────────────────────────────────────────
init(autoreset=True)
print(f"{Fore.YELLOW}{Style.BRIGHT}{'─'*60}")
print(f"{Fore.YELLOW}{Style.BRIGHT}   3D Printer ML Data Streamer — Mini Project")
print(f"{Fore.YELLOW}{Style.BRIGHT}{'─'*60}\n")

# ─────────────────────────────────────────────────────────────────
# SECTION 2: MongoDB Connection 
# ─────────────────────────────────────────────────────────────────
try:
    client = MongoClient(
        "mongodb+srv://databasesensor:1234mongo@navdeep.p32wk6a.mongodb.net/?appName=navdeep",
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")
    print(f"{Fore.GREEN}✅ MongoDB Connected!")
except errors.ConnectionFailure as e:
    print(f"{Fore.RED}❌ MongoDB Connection Failed: {e}")
    exit(1)

db         = client["printer_maintenance"]
collection = db["sensor_data_ml"]

# Auto open Atlas dashboard
webbrowser.open("https://cloud.mongodb.com")
print(f"{Fore.GREEN}🌐 MongoDB Atlas opened in browser.\n")

# ─────────────────────────────────────────────────────────────────
# SECTION 3: Physics-Based Failure Logic
# ─────────────────────────────────────────────────────────────────
def get_failure_status(air_t, proc_t, rpm, torque, wear, original_label):
    """
    Returns ALL active failure types for a row using physics rules.
    Multiple failures can be active at once (realistic).
    """
    active = []

    # HDF — Heat Dissipation Failure (loosened: temp diff < 10, rpm < 1500)
    if (proc_t - air_t) < 10.0 and rpm < 1500:
        active.append("HDF")

    # PWF — Power Failure (widened band: < 4500 or > 8500)
    power = torque * (rpm * 2 * np.pi / 60)
    if power < 4500 or power > 8500:
        active.append("PWF")

    # OSF — Overstrain Failure (lowered threshold: 9000)
    if (wear * torque) > 9000:
        active.append("OSF")

    # TWF — Tool Wear Failure (lowered: wear >= 170)
    if wear >= 170:
        active.append("TWF")

    # RNF — Random Failure (original label OR small random chance)
    if (original_label == 1 and len(active) == 0) or np.random.rand() < 0.015:
        active.append("RNF")

    if len(active) == 0:
        return "Normal", 0
    else:
        return ", ".join(active), 1


# ─────────────────────────────────────────────────────────────────
# SECTION 4: Load CSV & Validate
# ─────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]',
    'Tool wear [min]', 'Machine failure'
]

try:
    df = pd.read_csv('ai4i2020.csv')
except FileNotFoundError:
    print(f"{Fore.RED}❌ 'ai4i2020.csv' not found.")
    exit(1)

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    print(f"{Fore.RED}❌ Missing columns: {missing}")
    exit(1)

before = len(df)
df.dropna(subset=REQUIRED_COLS, inplace=True)
dropped = before - len(df)
if dropped:
    print(f"{Fore.YELLOW}⚠️  Dropped {dropped} rows with missing values.")

print(f"{Fore.GREEN}✅ CSV Loaded — {len(df)} rows ready.\n")

# ─────────────────────────────────────────────────────────────────
# SECTION 5: Build All Rows with Failure Injection
# ─────────────────────────────────────────────────────────────────
print(f"{Fore.YELLOW}🔧 Processing & injecting failure logic...")

all_rows   = []
type_count = {"HDF": 0, "PWF": 0, "OSF": 0, "TWF": 0, "RNF": 0, "Normal": 0}

for _, row in df.iterrows():
    air_t  = float(row['Air temperature [K]'])
    proc_t = float(row['Process temperature [K]'])
    rpm    = int(row['Rotational speed [rpm]'])
    torque = float(row['Torque [Nm]'])
    wear   = int(row['Tool wear [min]'])
    orig   = int(row['Machine failure'])

    status, label = get_failure_status(air_t, proc_t, rpm, torque, wear, orig)
    power         = round(torque * rpm * 2 * np.pi / 60, 2)

    all_rows.append({
        "timestamp" : datetime.now(),
        "air_temp"  : round(air_t, 2),
        "proc_temp" : round(proc_t, 2),
        "rpm"       : rpm,
        "torque"    : round(torque, 2),
        "wear"      : wear,
        "power_w"   : power,
        "status"    : status,
        "label"     : label,
    })

    # Count each type for summary
    if label == 0:
        type_count["Normal"] += 1
    else:
        for t in ["HDF", "PWF", "OSF", "TWF", "RNF"]:
            if t in status:
                type_count[t] += 1

# Shuffle — random order, no predictable failure pattern
random.seed(42)
random.shuffle(all_rows)

total   = len(all_rows)
f_count = sum(1 for r in all_rows if r['label'] == 1)
n_count = total - f_count

print(f"{Fore.CYAN}   Total Rows : {total}")
print(f"{Fore.CYAN}   Failures   : {f_count}  ({100*f_count/total:.1f}%)")
print(f"{Fore.CYAN}   Normals    : {n_count}  ({100*n_count/total:.1f}%)")
print(f"{Fore.WHITE}   ── Failure Breakdown ──")
for t, count in type_count.items():
    if t != "Normal":
        print(f"{Fore.MAGENTA}   {t}  : {count:>5}")
print(f"{Fore.GREEN}   Rows shuffled randomly ✔\n")

if total == 0:
    print(f"{Fore.RED}❌ No data found. Check your CSV.")
    exit(1)

# ─────────────────────────────────────────────────────────────────
# SECTION 6: Streaming Loop
# ─────────────────────────────────────────────────────────────────
print(f"{Fore.BLUE}{Style.BRIGHT}>>> Streaming Started — {total} rows...")
print(f"{Fore.BLUE}{'─'*60}\n")

success = 0
fail    = 0

for i, row_data in enumerate(all_rows):
    data = dict(row_data)
    data["timestamp"] = datetime.now()

    try:
        collection.insert_one(data)
        success += 1

        if data['label'] == 1:
            color = Fore.RED + Style.BRIGHT
            tag   = "FAILURE"
        else:
            color = Fore.CYAN
            tag   = "NORMAL "

        print(
            f"{color}[{tag}] "
            f"#{i+1:<5} | "
            f"Type: {data['status']:<20} | "
            f"RPM: {data['rpm']:>4} | "
            f"Torque: {data['torque']:>6.1f}Nm | "
            f"Wear: {data['wear']:>4}min | "
            f"Power: {data['power_w']:>8.1f}W"
        )

    except Exception as e:
        fail += 1
        print(f"{Fore.RED}❌ Insert error at row {i+1}: {e}")

    time.sleep(1)

# ─────────────────────────────────────────────────────────────────
# SECTION 7: Final Summary
# ─────────────────────────────────────────────────────────────────
print(f"\n{Fore.GREEN}{Style.BRIGHT}{'═'*60}")
print(f"{Fore.GREEN}{Style.BRIGHT}  ✅ Streaming Complete!")
print(f"{Fore.GREEN}  Total Streamed : {total} rows")
print(f"{Fore.GREEN}  Inserted       : {success} rows")
if fail:
    print(f"{Fore.RED}  Failed         : {fail} rows")
print(f"{Fore.CYAN}  Collection     : printer_maintenance → sensor_data_ml")
print(f"{Fore.WHITE}  Model training data ready")
print(f"{Fore.GREEN}{Style.BRIGHT}{'═'*60}")

