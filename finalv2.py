import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet


def create_pdf_report(eda_text, model_text):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("march_madness_report.pdf")
    content = []
    content.append(Paragraph("March Madness Upset Analysis", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Preformatted(eda_text, styles["Code"]))
    content.append(Spacer(1, 12))
    content.append(Preformatted(model_text, styles["Code"]))
    doc.build(content)


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------
BASE_DIR = Path("mmcsv")
MATCHUPS_FILE = BASE_DIR / "Tournament Matchups.csv"
KENPOM_FILE   = BASE_DIR / "KenPom Barttorvik.csv"

# CBB conference game logs live in a sibling folder called cbb_data/.
CBB_DIR = Path("cbb_data")


# --------------------------------------------------
# 1b. Style helpers
# --------------------------------------------------
def pretty_header(title):
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)


def pretty_df(df, title, n=10):
    pretty_header(title)
    print(df.head(n).to_string(index=False))


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
def load_data():
    matchups = pd.read_csv(MATCHUPS_FILE)
    kenpom   = pd.read_csv(KENPOM_FILE)
    return matchups, kenpom


# --------------------------------------------------
# 2b. [NEW] LOAD & NORMALIZE CBB REGULAR-SEASON LOGS
# --------------------------------------------------

# Hand-crafted mapping for team names that differ between the KenPom/Matchups
# naming convention and the CBB game-log naming convention.
# Left  = KenPom / Tournament Matchups name
# Right = CBB game-log name
_KENPOM_TO_CBB = {
    "Albany":                 "Albany (NY)",
    "Arkansas Pine Bluff":    "Arkansas-Pine Bluff",
    "BYU":                    "Brigham Young",
    "Cal Baptist":            "California Baptist",
    "Detroit":                "Detroit Mercy",
    "Fairleigh Dickinson":    "Fairleigh Dickinson",
    "Gardner Webb":           "Gardner-Webb",
    "Grambling St.":          "Grambling",
    "LIU Brooklyn":           "LIU Brooklyn",
    "LSU":                    "Louisiana State",
    "Louisiana Lafayette":    "Louisiana",
    "Loyola Chicago":         "Loyola (IL)",
    "McNeese St.":            "McNeese",
    "Nebraska Omaha":         "Nebraska Omaha",
    "North Carolina St.":     "North Carolina State",
    "Penn":                   "Pennsylvania",
    "Queens":                 "Queens (NC)",
    "SIU Edwardsville":       "SIU-Edwardsville",
    "SMU":                    "Southern Methodist",
    "Saint Francis":          "Saint Francis (PA)",
    "Sam Houston St.":        "Sam Houston",
    "Southern Miss":          "Southern Mississippi",
    "St. John's":             "St. John's (NY)",
    "TCU":                    "Texas Christian",
    "Texas A&M Corpus Chris": "Texas A&M-Corpus Christi",
    "UMBC":                   "Maryland-Baltimore County",
    "UNLV":                   "Nevada-Las Vegas",
    "USC":                    "Southern California",
    "VCU":                    "Virginia Commonwealth",
}


def _normalize_team_name(name: str) -> str:
    """Lower-case, expand abbreviations, strip punctuation for fuzzy matching."""
    name = str(name).strip().lower()
    name = name.replace("st.", "state").replace("st ", "state ")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_cbb_data() -> pd.DataFrame:
    """
    Read all per-year CBB conference game-log CSVs from CBB_DIR.
    Columns kept: season, home_team, away_team, home_score, away_score.
    """
    frames = []
    if not CBB_DIR.exists():
        print(f"WARNING: CBB data directory '{CBB_DIR}' not found. "
              "Head-to-head features will be all NaN.")
        return pd.DataFrame()

    for csv_path in sorted(CBB_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=[
                "season", "home_team", "away_team", "home_score", "away_score"
            ])
            frames.append(df)
        except Exception as e:
            print(f"WARNING: could not read {csv_path}: {e}")

    if not frames:
        return pd.DataFrame()

    cbb = pd.concat(frames, ignore_index=True)
    cbb = cbb.dropna(subset=["home_score", "away_score"])
    cbb["home_score"] = pd.to_numeric(cbb["home_score"], errors="coerce")
    cbb["away_score"] = pd.to_numeric(cbb["away_score"], errors="coerce")
    cbb = cbb.dropna(subset=["home_score", "away_score"])
    cbb["season"] = pd.to_numeric(cbb["season"], errors="coerce").astype("Int64")
    return cbb


def build_h2h_lookup(cbb: pd.DataFrame, all_tournament_teams: set) -> dict:
    """
    Build a dict: (season, teamA_kenpom, teamB_kenpom) -> list of game dicts.
    Both orderings are stored so lookups are symmetric and O(1).

    Each game dict contains:
        home_team, away_team, home_score, away_score, winner, margin
    """
    if cbb.empty:
        return {}

    cbb_teams_all = set(cbb["home_team"].unique()) | set(cbb["away_team"].unique())
    cbb_norm_map  = {_normalize_team_name(t): t for t in cbb_teams_all}

    # Build KenPom -> CBB name mapping
    kenpom_to_cbb = {}
    for kp_name in all_tournament_teams:
        if kp_name in _KENPOM_TO_CBB:
            kenpom_to_cbb[kp_name] = _KENPOM_TO_CBB[kp_name]
            continue
        norm = _normalize_team_name(kp_name)
        if norm in cbb_norm_map:
            kenpom_to_cbb[kp_name] = cbb_norm_map[norm]

    cbb_to_kenpom = {v: k for k, v in kenpom_to_cbb.items()}

    lookup = {}
    for _, row in cbb.iterrows():
        ht_kp = cbb_to_kenpom.get(row["home_team"])
        at_kp = cbb_to_kenpom.get(row["away_team"])
        if ht_kp is None or at_kp is None:
            continue

        hs, as_ = row["home_score"], row["away_score"]
        winner = ht_kp if hs > as_ else at_kp
        margin = abs(hs - as_)
        season = row["season"]

        game = {
            "home_team": ht_kp, "away_team": at_kp,
            "home_score": hs, "away_score": as_,
            "winner": winner, "margin": margin,
        }
        for key in [(season, ht_kp, at_kp), (season, at_kp, ht_kp)]:
            lookup.setdefault(key, []).append(game)

    return lookup


def compute_h2h_features(year: int, fav_team: str,
                          dog_team: str, lookup: dict) -> dict:
    """
    Return head-to-head regular-season features for one tournament matchup.

    Features returned:
        h2h_played        – 1 if they met in the regular season, else 0
        h2h_games         – number of regular-season meetings
        h2h_dog_wins      – wins by the underdog
        h2h_fav_wins      – wins by the favorite
        h2h_dog_win_flag  – 1 if underdog won more H2H games than favorite
        h2h_avg_margin    – signed avg margin (+ = underdog outperformed)
        h2h_dog_won_last  – 1 if underdog won the most-recent H2H game
    """
    games = lookup.get((year, fav_team, dog_team), [])

    if not games:
        return {
            "h2h_played": 0, "h2h_games": 0,
            "h2h_dog_wins": np.nan, "h2h_fav_wins": np.nan,
            "h2h_dog_win_flag": np.nan,
            "h2h_avg_margin": np.nan, "h2h_dog_won_last": np.nan,
        }

    dog_wins = sum(1 for g in games if g["winner"] == dog_team)
    fav_wins = len(games) - dog_wins

    signed_margins = [
        g["margin"] if g["winner"] == dog_team else -g["margin"]
        for g in games
    ]
    avg_margin = float(np.mean(signed_margins))

    last_game    = games[-1]  # CSVs are date-sorted; last = most recent
    dog_won_last = 1 if last_game["winner"] == dog_team else 0

    return {
        "h2h_played":       1,
        "h2h_games":        len(games),
        "h2h_dog_wins":     dog_wins,
        "h2h_fav_wins":     fav_wins,
        "h2h_dog_win_flag": int(dog_wins > fav_wins),
        "h2h_avg_margin":   avg_margin,
        "h2h_dog_won_last": dog_won_last,
    }


# --------------------------------------------------
# 3. CONVERT BRACKET ROWS INTO GAMES
# --------------------------------------------------
def build_games_from_matchups(matchups: pd.DataFrame) -> pd.DataFrame:
    df = matchups.copy()
    df = df[df["SCORE"].notna()].copy()
    df = df.sort_values(["YEAR", "CURRENT ROUND", "BY YEAR NO"],
                        ascending=[True, True, False]).reset_index(drop=True)

    games = []
    for (year, current_round), group in df.groupby(["YEAR", "CURRENT ROUND"], sort=True):
        group = group.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)
        if len(group) % 2 != 0:
            print(f"Warning: odd rows for YEAR={year}, CURRENT ROUND={current_round}. Skipping last.")

        for i in range(0, len(group) - 1, 2):
            team1 = group.iloc[i]
            team2 = group.iloc[i + 1]

            if team1["ROUND"] < team2["ROUND"]:
                winner, loser = team1, team2
            elif team2["ROUND"] < team1["ROUND"]:
                winner, loser = team2, team1
            else:
                if team1["SCORE"] > team2["SCORE"]:
                    winner, loser = team1, team2
                elif team2["SCORE"] > team1["SCORE"]:
                    winner, loser = team2, team1
                else:
                    continue

            if team1["SEED"] < team2["SEED"]:
                favorite, underdog = team1, team2
            elif team2["SEED"] < team1["SEED"]:
                favorite, underdog = team2, team1
            else:
                favorite, underdog = team1, team2

            upset = int(
                winner["TEAM NO"] == underdog["TEAM NO"]
                and underdog["SEED"] > favorite["SEED"]
            )

            games.append({
                "YEAR": year, "CURRENT_ROUND": current_round,
                "FavoriteTeamNo": favorite["TEAM NO"], "FavoriteTeam": favorite["TEAM"],
                "FavoriteSeed": favorite["SEED"],      "FavoriteScore": favorite["SCORE"],
                "UnderdogTeamNo": underdog["TEAM NO"], "UnderdogTeam": underdog["TEAM"],
                "UnderdogSeed": underdog["SEED"],      "UnderdogScore": underdog["SCORE"],
                "WinnerTeamNo": winner["TEAM NO"],     "WinnerTeam": winner["TEAM"],
                "LoserTeamNo": loser["TEAM NO"],       "LoserTeam": loser["TEAM"],
                "Upset": upset,
            })

    games_df = pd.DataFrame(games)
    games_df["SeedDiff"] = games_df["UnderdogSeed"] - games_df["FavoriteSeed"]
    return games_df


# --------------------------------------------------
# 4. SELECT TEAM FEATURES FROM KENPOM/BARTTORVIK
# --------------------------------------------------
def prep_team_features(kenpom: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "YEAR", "TEAM NO", "TEAM", "SEED",
        "KADJ EM", "KADJ O", "KADJ D",
        "BADJ EM", "BADJ O", "BADJ D",
        "BARTHAG", "GAMES", "W", "L", "WIN%",
        "EFG%", "EFG%D", "TOV%", "TOV%D",
        "OREB%", "DREB%", "2PT%", "2PT%D",
        "3PT%", "3PT%D", "AST%",
        "OP OREB%", "OP DREB%", "RAW T",
    ]
    keep = [c for c in feature_cols if c in kenpom.columns]
    return kenpom[keep].copy()


# --------------------------------------------------
# 5. MERGE TEAM FEATURES ONTO FAVORITE / UNDERDOG
# --------------------------------------------------
def merge_team_features(games_df: pd.DataFrame,
                        team_features: pd.DataFrame) -> pd.DataFrame:
    rename_template = {
        "TEAM NO":  "{role}TeamNo",  "TEAM":     "{role}TeamNameFromStats",
        "SEED":     "{role}SeedFromStats",
        "KADJ EM":  "{role}_KADJ_EM", "KADJ O":   "{role}_KADJ_O",
        "KADJ D":   "{role}_KADJ_D",  "BADJ EM":  "{role}_BADJ_EM",
        "BADJ O":   "{role}_BADJ_O",  "BADJ D":   "{role}_BADJ_D",
        "BARTHAG":  "{role}_BARTHAG", "GAMES":    "{role}_GAMES",
        "W":        "{role}_W",       "L":        "{role}_L",
        "WIN%":     "{role}_WIN_PCT", "EFG%":     "{role}_EFG",
        "EFG%D":    "{role}_EFGD",    "TOV%":     "{role}_TOV",
        "TOV%D":    "{role}_TOVD",    "OREB%":    "{role}_OREB",
        "DREB%":    "{role}_DREB",    "2PT%":     "{role}_2PT",
        "2PT%D":    "{role}_2PTD",    "3PT%":     "{role}_3PT",
        "3PT%D":    "{role}_3PTD",    "AST%":     "{role}_AST",
        "OP OREB%": "{role}_OP_OREB", "OP DREB%": "{role}_OP_DREB",
        "RAW T":    "{role}_RAW_T",
    }

    merged = games_df.copy()
    for role, join_key in [("Favorite", "FavoriteTeamNo"), ("Underdog", "UnderdogTeamNo")]:
        rmap = {k: v.format(role=role) for k, v in rename_template.items()
                if k in team_features.columns}
        side = team_features.rename(columns=rmap)
        merged = merged.merge(side, on=["YEAR", join_key], how="left")
    return merged


# --------------------------------------------------
# 6. CREATE DIFFERENCE FEATURES
# --------------------------------------------------
def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diff_pairs = [
        ("KADJ_EM_Diff",  "Underdog_KADJ_EM",  "Favorite_KADJ_EM"),
        ("KADJ_O_Diff",   "Underdog_KADJ_O",   "Favorite_KADJ_O"),
        ("KADJ_D_Diff",   "Underdog_KADJ_D",   "Favorite_KADJ_D"),
        ("BADJ_EM_Diff",  "Underdog_BADJ_EM",  "Favorite_BADJ_EM"),
        ("BARTHAG_Diff",  "Underdog_BARTHAG",  "Favorite_BARTHAG"),
        ("WIN_PCT_Diff",  "Underdog_WIN_PCT",  "Favorite_WIN_PCT"),
        ("EFG_Diff",      "Underdog_EFG",      "Favorite_EFG"),
        ("EFGD_Diff",     "Underdog_EFGD",     "Favorite_EFGD"),
        ("TOV_Diff",      "Underdog_TOV",      "Favorite_TOV"),
        ("TOVD_Diff",     "Underdog_TOVD",     "Favorite_TOVD"),
        ("OREB_Diff",     "Underdog_OREB",     "Favorite_OREB"),
        ("DREB_Diff",     "Underdog_DREB",     "Favorite_DREB"),
        ("TWO_PT_Diff",   "Underdog_2PT",      "Favorite_2PT"),
        ("THREE_PT_Diff", "Underdog_3PT",      "Favorite_3PT"),
        ("AST_Diff",      "Underdog_AST",      "Favorite_AST"),
    ]
    for new_col, dog_col, fav_col in diff_pairs:
        if dog_col in df.columns and fav_col in df.columns:
            df[new_col] = df[dog_col] - df[fav_col]
    return df


# --------------------------------------------------
# 6b. [NEW] ADD HEAD-TO-HEAD REGULAR-SEASON FEATURES
# --------------------------------------------------
def add_h2h_features(games_df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """
    For every tournament matchup, look up whether the two teams played
    in the regular season that same year and attach summary features.

    New columns added:
        h2h_played        - 1 if they met in regular season, else 0
        h2h_games         - number of regular-season meetings
        h2h_dog_wins      - underdog wins in reg-season H2H
        h2h_fav_wins      - favorite wins in reg-season H2H
        h2h_dog_win_flag  - 1 if underdog won more H2H games than favorite
        h2h_avg_margin    - signed avg margin (+ = underdog advantage)
        h2h_dog_won_last  - 1 if underdog won the most-recent H2H game
    """
    if not lookup:
        for col in ["h2h_played", "h2h_games", "h2h_dog_wins", "h2h_fav_wins",
                    "h2h_dog_win_flag", "h2h_avg_margin", "h2h_dog_won_last"]:
            games_df[col] = np.nan
        return games_df

    h2h_rows = games_df.apply(
        lambda row: compute_h2h_features(
            row["YEAR"], row["FavoriteTeam"], row["UnderdogTeam"], lookup
        ),
        axis=1, result_type="expand",
    )
    return pd.concat([games_df.reset_index(drop=True),
                      h2h_rows.reset_index(drop=True)], axis=1)


# --------------------------------------------------
# 7. QUICK EDA
# --------------------------------------------------
def basic_upset_analysis(df: pd.DataFrame):
    output = []

    output.append("OVERALL STATS")
    output.append(f"Total games: {len(df)}")
    output.append(f"Upset rate: {df['Upset'].mean():.3f} ({df['Upset'].mean()*100:.1f}%)\n")

    round_df = df.groupby("CURRENT_ROUND")["Upset"].agg(["count", "mean"])
    round_df["mean"] = (round_df["mean"] * 100).round(1)
    output.append("UPSETS BY ROUND")
    output.append(round_df.to_string())
    output.append("")

    seed_df = df.groupby("SeedDiff")["Upset"].agg(["count", "mean"])
    seed_df["mean"] = (seed_df["mean"] * 100).round(1)
    output.append("UPSETS BY SEED DIFFERENCE")
    output.append(seed_df.to_string())
    output.append("")

    # [NEW] Head-to-head summary
    if "h2h_played" in df.columns:
        h2h_games   = df[df["h2h_played"] == 1]
        no_h2h      = df[df["h2h_played"] == 0]
        output.append("HEAD-TO-HEAD REGULAR SEASON SUMMARY")
        output.append(
            f"Tournament matchups with a reg-season meeting: "
            f"{len(h2h_games)} / {len(df)} ({len(h2h_games)/len(df)*100:.1f}%)"
        )
        if len(h2h_games) > 0:
            output.append(f"Upset rate when H2H exists:    {h2h_games['Upset'].mean()*100:.1f}%")
            output.append(f"Upset rate when no H2H exists: {no_h2h['Upset'].mean()*100:.1f}%")

            dog_better = h2h_games[h2h_games["h2h_dog_win_flag"] == 1]
            fav_better = h2h_games[h2h_games["h2h_dog_win_flag"] == 0]
            output.append(
                f"\nUpset rate when underdog had better H2H record: "
                f"{dog_better['Upset'].mean()*100:.1f}% (n={len(dog_better)})"
            )
            output.append(
                f"Upset rate when favorite had better H2H record:  "
                f"{fav_better['Upset'].mean()*100:.1f}% (n={len(fav_better)})"
            )
        output.append("")

    for line in output:
        print(line)
    return "\n".join(output)


# --------------------------------------------------
# 8. MODEL
# --------------------------------------------------
def build_model(df: pd.DataFrame):
    model_df = df.copy()

    feature_cols = [
        # Original rating / stat difference features
        "SeedDiff",
        "KADJ_EM_Diff", "KADJ_O_Diff", "KADJ_D_Diff",
        "BADJ_EM_Diff", "BARTHAG_Diff",
        "WIN_PCT_Diff", "EFG_Diff", "EFGD_Diff",
        "TOV_Diff", "TOVD_Diff", "OREB_Diff", "DREB_Diff",
        "TWO_PT_Diff", "THREE_PT_Diff", "AST_Diff",
        # [NEW] Head-to-head regular-season features
        "h2h_played",        # did they meet this season? (0/1)
        "h2h_dog_win_flag",  # underdog had better H2H record? (0/1/NaN)
        "h2h_avg_margin",    # signed avg margin (+ = underdog advantage)
        "h2h_dog_won_last",  # underdog won the most-recent H2H game? (0/1/NaN)
    ]

    feature_cols = [c for c in feature_cols if c in model_df.columns]

    X = model_df[feature_cols]
    y = model_df["Upset"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds)
    cm     = confusion_matrix(y_test, preds)
    auc    = roc_auc_score(y_test, probs)

    print("\nClassification report:")
    print(report)
    print("\nConfusion matrix:")
    print(cm)
    print("\nROC-AUC:")
    print(auc)

    coefs   = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "Feature": feature_cols, "Coefficient": coefs
    }).sort_values("Coefficient", key=np.abs, ascending=False)

    print("\nFeature coefficients:")
    print(coef_df)

    model_text = f"""
MODEL PERFORMANCE

Classification Report:
{report}

Confusion Matrix:
{cm}

ROC-AUC:
{auc:.3f}

Feature Coefficients:
{coef_df.to_string(index=False)}
"""
    return pipeline, coef_df, y_test, preds, probs, model_text


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------
def main():
    matchups, kenpom = load_data()

    print("Tournament Matchups columns:")
    print(matchups.columns.tolist())

    print("\nKenPom Barttorvik columns (first 40):")
    print(kenpom.columns.tolist()[:40])

    games_df      = build_games_from_matchups(matchups)
    team_features = prep_team_features(kenpom)
    merged        = merge_team_features(games_df, team_features)
    merged        = add_difference_features(merged)

    # Add conference info
    team_with_conf = kenpom[["TEAM", "CONF", "YEAR"]]
    merged = merged.merge(team_with_conf,
                          left_on=["YEAR", "UnderdogTeam"],
                          right_on=["YEAR", "TEAM"], how="left")
    merged = merged.rename(columns={"CONF": "UnderdogConf"}).drop(columns=["TEAM"])

    # --------------------------------------------------
    # [NEW] LOAD CBB DATA AND ADD HEAD-TO-HEAD FEATURES
    # --------------------------------------------------
    print("\nLoading CBB regular-season game logs...")
    cbb = load_cbb_data()

    if not cbb.empty:
        print(f"  Loaded {len(cbb):,} regular-season games "
              f"across seasons {cbb['season'].min()}–{cbb['season'].max()}")
        all_tournament_teams = (
            set(merged["FavoriteTeam"].dropna()) | set(merged["UnderdogTeam"].dropna())
        )
        lookup = build_h2h_lookup(cbb, all_tournament_teams)
        matched_pairs = len(lookup) // 2
        print(f"  Built H2H lookup with {matched_pairs:,} unique team pairs.")
    else:
        lookup = {}

    merged = add_h2h_features(merged, lookup)

    if "h2h_played" in merged.columns:
        n_h2h = int(merged["h2h_played"].sum())
        print(f"\n  {n_h2h} tournament matchups had a regular-season meeting "
              f"({n_h2h/len(merged)*100:.1f}%)")

    print("\nGames preview:")
    print(games_df.head())

    print("\nMerged preview (H2H columns):")
    preview_cols = [c for c in [
        "YEAR", "FavoriteTeam", "UnderdogTeam", "Upset",
        "h2h_played", "h2h_games", "h2h_dog_win_flag",
        "h2h_avg_margin", "h2h_dog_won_last"
    ] if c in merged.columns]
    print(merged[preview_cols].head(10).to_string(index=False))

    eda_text = basic_upset_analysis(merged)
    model, coef_df, y_test, preds, probs, model_text = build_model(merged)

    create_pdf_report(eda_text, model_text)

    return merged, model, coef_df


if __name__ == "__main__":
    merged_data, model, coefficients = main()