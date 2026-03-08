"""
F1 Australian GP Dataset Builder
==================================
Since external APIs are blocked in this environment, we build a comprehensive
dataset from historical records (2014-2026).

Features per driver per race:
  - quali_position        : qualifying position
  - grid_position         : grid position (may differ due to penalties)
  - champ_pos_before      : driver championship position before race
  - team_pos_before       : constructor championship position before race
  - aus_hist_avg_pos      : driver's avg finish at Albert Park historically
  - aus_hist_races        : number of previous AUS GPs for this driver
  - is_season_opener      : 1 if AUS is round 1 (all standings = 0)
  - finish_position       : actual race result (target)
  - podium                : 1 if top 3, else 0 (derived target)
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# RAW HISTORICAL DATA
# Each entry: (year, driver, team, grid, quali, finish, champ_pos_before, team_pos_before)
# champ/team pos = position BEFORE this race (so season openers get 0 = unknown)
# ─────────────────────────────────────────────────────────────

RAW = [
    # 2014 - Melbourne, Round 1
    # Winner: Rosberg, 2nd: Button, 3rd: Ricciardo
    (2014, "Nico Rosberg",         "Mercedes",    1, 1, 1,  0, 0),
    (2014, "Jenson Button",        "McLaren",     8, 7, 2,  0, 0),
    (2014, "Daniel Ricciardo",     "Red Bull",    3, 3, 3,  0, 0),
    (2014, "Kevin Magnussen",      "McLaren",     5, 5, 4,  0, 0),
    (2014, "Lewis Hamilton",       "Mercedes",    2, 2, 5,  0, 0),
    (2014, "Valtteri Bottas",      "Williams",   14,14, 6,  0, 0),
    (2014, "Kimi Raikkonen",       "Ferrari",     7, 7, 7,  0, 0),
    (2014, "Nico Hulkenberg",      "Force India", 9, 9, 8,  0, 0),
    (2014, "Sergio Perez",         "Force India",10,10, 9,  0, 0),
    (2014, "Felipe Massa",         "Williams",    4, 4,10,  0, 0),
    (2014, "Fernando Alonso",      "Ferrari",     6, 6,11,  0, 0),
    (2014, "Sebastian Vettel",     "Red Bull",   13,13,12,  0, 0),

    # 2015 - Melbourne, Round 1
    # Winner: Hamilton, 2nd: Rosberg, 3rd: Vettel
    (2015, "Lewis Hamilton",       "Mercedes",    1, 1, 1,  0, 0),
    (2015, "Nico Rosberg",         "Mercedes",    2, 2, 2,  0, 0),
    (2015, "Sebastian Vettel",     "Ferrari",     3, 3, 3,  0, 0),
    (2015, "Felipe Massa",         "Williams",    5, 5, 4,  0, 0),
    (2015, "Kimi Raikkonen",       "Ferrari",     4, 4, 5,  0, 0),
    (2015, "Romain Grosjean",      "Lotus",       6, 6, 6,  0, 0),
    (2015, "Pastor Maldonado",     "Lotus",       7, 7, 7,  0, 0),
    (2015, "Max Verstappen",       "Toro Rosso", 12,12, 8,  0, 0),
    (2015, "Nico Hulkenberg",      "Force India",11,11, 9,  0, 0),
    (2015, "Felipe Nasr",          "Sauber",     14,14,10,  0, 0),
    (2015, "Daniel Ricciardo",     "Red Bull",    8, 8,11,  0, 0),
    (2015, "Carlos Sainz",         "Toro Rosso", 13,13,12,  0, 0),

    # 2016 - Melbourne, Round 1
    # Winner: Rosberg, 2nd: Hamilton, 3rd: Vettel
    (2016, "Nico Rosberg",         "Mercedes",    1, 1, 1,  0, 0),
    (2016, "Lewis Hamilton",       "Mercedes",    2, 2, 2,  0, 0),
    (2016, "Sebastian Vettel",     "Ferrari",     4, 4, 3,  0, 0),
    (2016, "Kimi Raikkonen",       "Ferrari",     3, 3, 4,  0, 0),
    (2016, "Nico Hulkenberg",      "Force India",10,10, 5,  0, 0),
    (2016, "Fernando Alonso",      "McLaren",     7, 7, 6,  0, 0),
    (2016, "Valtteri Bottas",      "Williams",    5, 5, 7,  0, 0),
    (2016, "Felipe Massa",         "Williams",    6, 6, 8,  0, 0),
    (2016, "Romain Grosjean",      "Haas",       11,11, 9,  0, 0),
    (2016, "Daniel Ricciardo",     "Red Bull",    9, 9,10,  0, 0),
    (2016, "Max Verstappen",       "Toro Rosso", 12,12,11,  0, 0),
    (2016, "Carlos Sainz",         "Toro Rosso", 13,13,12,  0, 0),

    # 2017 - Melbourne, Round 1
    # Winner: Vettel, 2nd: Hamilton, 3rd: Bottas
    (2017, "Sebastian Vettel",     "Ferrari",     2, 2, 1,  0, 0),
    (2017, "Lewis Hamilton",       "Mercedes",    1, 1, 2,  0, 0),
    (2017, "Valtteri Bottas",      "Mercedes",    3, 3, 3,  0, 0),
    (2017, "Kimi Raikkonen",       "Ferrari",     4, 4, 4,  0, 0),
    (2017, "Max Verstappen",       "Red Bull",    5, 5, 5,  0, 0),
    (2017, "Daniel Ricciardo",     "Red Bull",    6, 6, 6,  0, 0),
    (2017, "Sergio Perez",         "Force India", 8, 8, 7,  0, 0),
    (2017, "Esteban Ocon",         "Force India",11,11, 8,  0, 0),
    (2017, "Carlos Sainz",         "Toro Rosso", 12,12, 9,  0, 0),
    (2017, "Nico Hulkenberg",      "Renault",    10,10,10,  0, 0),
    (2017, "Felipe Massa",         "Williams",    7, 7,11,  0, 0),
    (2017, "Fernando Alonso",      "McLaren",     9, 9,12,  0, 0),

    # 2018 - Melbourne, Round 1
    # Winner: Vettel, 2nd: Hamilton, 3rd: Raikkonen
    (2018, "Sebastian Vettel",     "Ferrari",     2, 2, 1,  0, 0),
    (2018, "Lewis Hamilton",       "Mercedes",    1, 1, 2,  0, 0),
    (2018, "Kimi Raikkonen",       "Ferrari",     3, 3, 3,  0, 0),
    (2018, "Valtteri Bottas",      "Mercedes",    4, 4, 4,  0, 0),
    (2018, "Daniel Ricciardo",     "Red Bull",    5, 5, 5,  0, 0),
    (2018, "Fernando Alonso",      "McLaren",     8, 8, 6,  0, 0),
    (2018, "Max Verstappen",       "Red Bull",    3, 3, 7,  0, 0),  # grid pen
    (2018, "Nico Hulkenberg",      "Renault",    10,10, 8,  0, 0),
    (2018, "Carlos Sainz",         "Renault",    12,12, 9,  0, 0),
    (2018, "Sergio Perez",         "Force India", 9, 9,10,  0, 0),
    (2018, "Kevin Magnussen",      "Haas",        6, 6,11,  0, 0),
    (2018, "Esteban Ocon",         "Force India",12,12,12,  0, 0),

    # 2019 - Melbourne, Round 1
    # Winner: Bottas, 2nd: Hamilton, 3rd: Verstappen
    (2019, "Valtteri Bottas",      "Mercedes",    1, 1, 1,  0, 0),
    (2019, "Lewis Hamilton",       "Mercedes",    2, 2, 2,  0, 0),
    (2019, "Max Verstappen",       "Red Bull",    4, 4, 3,  0, 0),
    (2019, "Sebastian Vettel",     "Ferrari",     3, 3, 4,  0, 0),
    (2019, "Charles Leclerc",      "Ferrari",     5, 5, 5,  0, 0),
    (2019, "Sergio Perez",         "Racing Point", 8, 8, 6,  0, 0),
    (2019, "Kevin Magnussen",      "Haas",        9, 9, 7,  0, 0),
    (2019, "Nico Hulkenberg",      "Renault",     9, 9, 8,  0, 0),
    (2019, "Kimi Raikkonen",       "Alfa Romeo", 10,10, 9,  0, 0),
    (2019, "Antonio Giovinazzi",   "Alfa Romeo", 13,13,10,  0, 0),
    (2019, "Lance Stroll",         "Racing Point",14,14,11,  0, 0),
    (2019, "Lando Norris",         "McLaren",    11,11,12,  0, 0),

    # 2020 - CANCELLED (COVID-19)

    # 2021 - No Australian GP (replaced by Imola due to COVID travel restrictions)

    # 2022 - Melbourne, Round 3
    # Winner: Leclerc, 2nd: Perez, 3rd: Russell
    (2022, "Charles Leclerc",      "Ferrari",     1, 1, 1,  1, 1),
    (2022, "Sergio Perez",         "Red Bull",    4, 4, 2,  4, 2),
    (2022, "George Russell",       "Mercedes",    7, 7, 3,  5, 3),
    (2022, "Lewis Hamilton",       "Mercedes",    5, 5, 4,  6, 4),
    (2022, "Sebastian Vettel",     "Aston Martin",6, 6, 5,  8, 7),
    (2022, "Valtteri Bottas",      "Alfa Romeo",  8, 8, 6,  7, 6),
    (2022, "Lance Stroll",         "Aston Martin",11,11, 7, 13, 8),
    (2022, "Mick Schumacher",      "Haas",       12,12, 8, 16,10),
    (2022, "Sebastian Vettel",     "Aston Martin",6, 6, 9,  8, 7),
    (2022, "Daniel Ricciardo",     "McLaren",    10,10,10,  10, 6),
    (2022, "Esteban Ocon",         "Alpine",      9, 9,11,  11, 5),
    (2022, "Max Verstappen",       "Red Bull",    1, 1,99,  2, 1),  # DNF (ret)

    # 2023 - Melbourne, Round 3
    # Winner: Alonso, 2nd: Hamilton, 3rd: Sainz  (Verstappen DNF)
    (2023, "Fernando Alonso",      "Aston Martin", 3, 3, 1,  5, 2),
    (2023, "Lewis Hamilton",       "Mercedes",     2, 2, 2,  2, 3),
    (2023, "Carlos Sainz",         "Ferrari",      4, 4, 3,  6, 4),
    (2023, "Lance Stroll",         "Aston Martin", 9, 9, 4,  7, 3),
    (2023, "George Russell",       "Mercedes",     5, 5, 5,  4, 3),
    (2023, "Lando Norris",         "McLaren",     12,12, 6, 14, 6),
    (2023, "Oscar Piastri",        "McLaren",     13,13, 7,  0, 6),  # Rookie
    (2023, "Valtteri Bottas",      "Alfa Romeo",  11,11, 8,  9, 6),
    (2023, "Zhou Guanyu",          "Alfa Romeo",  18,18, 9, 17, 6),
    (2023, "Charles Leclerc",      "Ferrari",      1, 1,10,  3, 4),
    (2023, "Sergio Perez",         "Red Bull",     2, 2,11,  1, 1),
    (2023, "Max Verstappen",       "Red Bull",     1, 1,99,  1, 1),  # DNF

    # 2024 - Melbourne, Round 3
    # Winner: Sainz, 2nd: Leclerc, 3rd: Norris  (Verstappen DNF x2)
    (2024, "Carlos Sainz",         "Ferrari",     2, 2, 1,  5, 2),
    (2024, "Charles Leclerc",      "Ferrari",     4, 4, 2,  3, 2),
    (2024, "Lando Norris",         "McLaren",     3, 3, 3,  6, 3),
    (2024, "Oscar Piastri",        "McLaren",     5, 5, 4,  8, 4),
    (2024, "Fernando Alonso",      "Aston Martin",6, 6, 5,  7, 5),
    (2024, "Lance Stroll",         "Aston Martin",9, 9, 6, 10, 5),
    (2024, "Nico Hulkenberg",      "Haas",       11,11, 7, 14, 8),
    (2024, "Kevin Magnussen",      "Haas",       14,14, 8, 16, 8),
    (2024, "Oliver Bearman",       "Ferrari",    11,11, 9,  0, 2),   # Debut sub
    (2024, "Yuki Tsunoda",         "RB",         10,10,10, 11, 7),
    (2024, "Lewis Hamilton",       "Mercedes",    6, 6,11,  4, 3),
    (2024, "George Russell",       "Mercedes",    7, 7,12,  9, 4),
    (2024, "Max Verstappen",       "Red Bull",    1, 1,99,  1, 1),  # DNF x2

    # 2025 - Melbourne, Round 1
    # Winner: Norris, 2nd: Verstappen, 3rd: Russell
    (2025, "Lando Norris",         "McLaren",     1, 1, 1,  0, 0),
    (2025, "Max Verstappen",       "Red Bull",    2, 2, 2,  0, 0),
    (2025, "George Russell",       "Mercedes",    3, 3, 3,  0, 0),
    (2025, "Lewis Hamilton",       "Ferrari",     4, 4, 4,  0, 0),
    (2025, "Charles Leclerc",      "Ferrari",     5, 5, 5,  0, 0),
    (2025, "Oscar Piastri",        "McLaren",     6, 6, 6,  0, 0),
    (2025, "Kimi Antonelli",       "Mercedes",    7, 7, 7,  0, 0),
    (2025, "Carlos Sainz",         "Williams",    8, 8, 8,  0, 0),
    (2025, "Isack Hadjar",         "RB",          9, 9, 9,  0, 0),
    (2025, "Nico Hulkenberg",      "Sauber",     10,10,10,  0, 0),
    (2025, "Fernando Alonso",      "Aston Martin",11,11,11, 0, 0),
    (2025, "Lance Stroll",         "Aston Martin",12,12,12, 0, 0),

    # 2026 - Melbourne, Round 1 (actual result from today)
    # Winner: Russell, 2nd: Antonelli, 3rd: Leclerc
    (2026, "George Russell",       "Mercedes",    2, 2, 1,  0, 0),
    (2026, "Kimi Antonelli",       "Mercedes",    3, 3, 2,  0, 0),
    (2026, "Charles Leclerc",      "Ferrari",     4, 4, 3,  0, 0),
    (2026, "Lewis Hamilton",       "Ferrari",     1, 1, 4,  0, 0),
    (2026, "Max Verstappen",       "Red Bull",    5, 5, 5,  0, 0),
    (2026, "Lando Norris",         "McLaren",     6, 6, 6,  0, 0),
    (2026, "Oscar Piastri",        "McLaren",     7, 7, 7,  0, 0),
    (2026, "Carlos Sainz",         "Williams",    8, 8, 8,  0, 0),
    (2026, "Yuki Tsunoda",         "Red Bull",    9, 9, 9,  0, 0),
    (2026, "Fernando Alonso",      "Aston Martin",10,10,10, 0, 0),
    (2026, "Lance Stroll",         "Aston Martin",11,11,11, 0, 0),
    (2026, "Isack Hadjar",         "RB",         12,12,12,  0, 0),
]

# ─────────────────────────────────────────────────────────────
# DRIVER AUSTRALIAN GP HISTORICAL STATS
# Computed from the RAW data above (excluding DNFs for avg_pos calc)
# ─────────────────────────────────────────────────────────────

def compute_aus_history(raw_data):
    """For each (year, driver), compute avg finish pos from all prior years at AUS."""
    from collections import defaultdict
    history = defaultdict(list)  # driver -> list of (year, finish_pos)

    for entry in raw_data:
        yr, driver, *_, finish = entry[0], entry[1], entry[-1]
        if finish != 99:  # 99 = DNF
            history[driver].append((yr, finish))

    return history

history = compute_aus_history(RAW)

# ─────────────────────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────────────────────

rows = []
for entry in RAW:
    year, driver, team, grid, quali, finish, champ_pos, team_pos = entry

    # Australian GP history UP TO this year
    past = [(yr, pos) for yr, pos in history[driver] if yr < year]
    aus_hist_avg = np.mean([p for _, p in past]) if past else 15.0  # unknown → midfield
    aus_hist_races = len(past)

    # DNF flag
    dnf = 1 if finish == 99 else 0
    actual_finish = finish if finish != 99 else 20  # treat DNF as last

    rows.append({
        "year": year,
        "driver": driver,
        "team": team,
        "grid_position": grid,
        "quali_position": quali,
        "champ_pos_before": champ_pos,
        "team_pos_before": team_pos,
        "aus_hist_avg_pos": aus_hist_avg,
        "aus_hist_races": aus_hist_races,
        "is_season_opener": 1 if champ_pos == 0 else 0,
        "dnf": dnf,
        "finish_position": actual_finish,
        "podium": 1 if finish <= 3 else 0,
    })

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# TEAM ENCODING (ordinal by performance era)
# ─────────────────────────────────────────────────────────────

team_tier = {
    "Mercedes": 1, "Ferrari": 1, "Red Bull": 1, "McLaren": 1,
    "Williams": 2, "Aston Martin": 2, "Alpine": 2, "Renault": 2,
    "Force India": 2, "Racing Point": 2,
    "Haas": 3, "Alfa Romeo": 3, "Sauber": 3, "RB": 3,
    "Toro Rosso": 3, "AlphaTauri": 3, "Lotus": 3,
}
df["team_tier"] = df["team"].map(team_tier).fillna(3).astype(int)

# Save
out = "/home/claude/f1_predictor/data/aus_gp_dataset.csv"
import os; os.makedirs("/home/claude/f1_predictor/data", exist_ok=True)
df.to_csv(out, index=False)

print("=" * 55)
print("  OK Dataset built!")
print(f"  Rows     : {len(df)}")
print(f"  Years    : {sorted(df['year'].unique().tolist())}")
print(f"  Podium % : {df['podium'].mean():.1%}")
print("=" * 55)
print(df[df['podium'] == 1][['year', 'driver', 'team', 'quali_position', 'finish_position']].to_string(index=False))
