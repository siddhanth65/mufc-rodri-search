import soccerdata as sd
import pandas as pd

fbref = sd.FBref(leagues="ENG-Premier League", seasons="2025")
df = fbref.read_player_season_stats(stat_type="standard")
df.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df.columns]
df = df.reset_index()

# Print every column so we know what we're working with
for c in df.columns:
    print(repr(c))