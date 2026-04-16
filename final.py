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

"""
CS152 Final Project Starter
Built to match your uploaded files:
- Tournament Matchups.csv
- KenPom Barttorvik.csv

What this code does:
1. Loads the tournament matchup bracket data
2. Converts every two rows into one tournament game
3. Identifies the favorite and underdog by seed
4. Creates the target variable: upset = 1 if the underdog wins
5. Merges in KenPom/Barttorvik team features
6. Builds a first logistic regression model to predict upsets

This follows the project plan's goals of identifying upset games, creating predictors,
and building a classification model. fileciteturn0file0L8-L23
"""

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------
# Put your CSVs in a folder next to this code file.
# Change "data" to your actual folder name if needed.
BASE_DIR = Path("data")
MATCHUPS_FILE = BASE_DIR / "Tournament Matchups.csv"
KENPOM_FILE = BASE_DIR / "KenPom Barttorvik.csv"


# --------------------------------------------------
# 1b. style
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
    kenpom = pd.read_csv(KENPOM_FILE)
    return matchups, kenpom


# --------------------------------------------------
# 3. CONVERT BRACKET ROWS INTO GAMES
# --------------------------------------------------
def build_games_from_matchups(matchups: pd.DataFrame) -> pd.DataFrame:
    """
    Tournament Matchups.csv is stored as one row per team entry in the bracket.
    Every pair of rows within the same year/current round corresponds to one game.

    Important columns observed in your file:
    - YEAR
    - BY YEAR NO
    - TEAM NO
    - TEAM
    - SEED
    - ROUND
    - CURRENT ROUND
    - SCORE

    ROUND appears to represent how far the team advanced overall.
    For a given CURRENT ROUND game:
    - the winner usually has ROUND < CURRENT ROUND
    - the loser usually has ROUND == CURRENT ROUND
    """

    df = matchups.copy()

    # Remove 2026 because many scores/results are incomplete in the uploaded file.
    df = df[df["SCORE"].notna()].copy()

    # Sort so bracket pairs stay together.
    # In your file, pairs appear consecutively when sorted by YEAR, CURRENT ROUND,
    # and BY YEAR NO descending.
    df = df.sort_values(["YEAR", "CURRENT ROUND", "BY YEAR NO"], ascending=[True, True, False]).reset_index(drop=True)

    games = []

    for (year, current_round), group in df.groupby(["YEAR", "CURRENT ROUND"], sort=True):
        group = group.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)

        if len(group) % 2 != 0:
            print(f"Warning: odd number of rows for YEAR={year}, CURRENT ROUND={current_round}. Last row will be skipped.")

        for i in range(0, len(group) - 1, 2):
            team1 = group.iloc[i]
            team2 = group.iloc[i + 1]

            # Determine winner.
            # Primary rule: lower ROUND value means team advanced farther.
            # Backup rule: higher score wins.
            if team1["ROUND"] < team2["ROUND"]:
                winner = team1
                loser = team2
            elif team2["ROUND"] < team1["ROUND"]:
                winner = team2
                loser = team1
            else:
                # Fallback to score if ROUND is tied
                if team1["SCORE"] > team2["SCORE"]:
                    winner = team1
                    loser = team2
                elif team2["SCORE"] > team1["SCORE"]:
                    winner = team2
                    loser = team1
                else:
                    # Rare/unexpected tie case; skip
                    continue

            # Favorite = lower seed number
            # Underdog = higher seed number
            if team1["SEED"] < team2["SEED"]:
                favorite = team1
                underdog = team2
            elif team2["SEED"] < team1["SEED"]:
                favorite = team2
                underdog = team1
            else:
                # Equal seeds are not true upset opportunities.
                # Keep them if you want, but upset target will be 0.
                # For now we keep them.
                favorite = team1
                underdog = team2

            upset = int(winner["TEAM NO"] == underdog["TEAM NO"] and underdog["SEED"] > favorite["SEED"])

            games.append({
                "YEAR": year,
                "CURRENT_ROUND": current_round,

                "FavoriteTeamNo": favorite["TEAM NO"],
                "FavoriteTeam": favorite["TEAM"],
                "FavoriteSeed": favorite["SEED"],
                "FavoriteScore": favorite["SCORE"],

                "UnderdogTeamNo": underdog["TEAM NO"],
                "UnderdogTeam": underdog["TEAM"],
                "UnderdogSeed": underdog["SEED"],
                "UnderdogScore": underdog["SCORE"],

                "WinnerTeamNo": winner["TEAM NO"],
                "WinnerTeam": winner["TEAM"],
                "LoserTeamNo": loser["TEAM NO"],
                "LoserTeam": loser["TEAM"],

                "Upset": upset
            })

    games_df = pd.DataFrame(games)
    games_df["SeedDiff"] = games_df["UnderdogSeed"] - games_df["FavoriteSeed"]
    return games_df


# --------------------------------------------------
# 4. SELECT TEAM FEATURES FROM KENPOM/BARTTORVIK
# --------------------------------------------------
def prep_team_features(kenpom: pd.DataFrame) -> pd.DataFrame:
    """
    Select a manageable set of team-level predictors from your KenPom Barttorvik file.
    These columns are present in the uploaded CSV.
    """

    feature_cols = [
        "YEAR",
        "TEAM NO",
        "TEAM",
        "SEED",
        "KADJ EM",
        "KADJ O",
        "KADJ D",
        "BADJ EM",
        "BADJ O",
        "BADJ D",
        "BARTHAG",
        "GAMES",
        "W",
        "L",
        "WIN%",
        "EFG%",
        "EFG%D",
        "TOV%",
        "TOV%D",
        "OREB%",
        "DREB%",
        "2PT%",
        "2PT%D",
        "3PT%",
        "3PT%D",
        "AST%",
        "OP OREB%",
        "OP DREB%",
        "RAW T",
    ]

    keep = [c for c in feature_cols if c in kenpom.columns]
    team_features = kenpom[keep].copy()
    return team_features


# --------------------------------------------------
# 5. MERGE TEAM FEATURES ONTO FAVORITE / UNDERDOG
# --------------------------------------------------
def merge_team_features(games_df: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    fav = team_features.rename(columns={
        "TEAM NO": "FavoriteTeamNo",
        "TEAM": "FavoriteTeamNameFromStats",
        "SEED": "FavoriteSeedFromStats",
        "KADJ EM": "Favorite_KADJ_EM",
        "KADJ O": "Favorite_KADJ_O",
        "KADJ D": "Favorite_KADJ_D",
        "BADJ EM": "Favorite_BADJ_EM",
        "BADJ O": "Favorite_BADJ_O",
        "BADJ D": "Favorite_BADJ_D",
        "BARTHAG": "Favorite_BARTHAG",
        "GAMES": "Favorite_GAMES",
        "W": "Favorite_W",
        "L": "Favorite_L",
        "WIN%": "Favorite_WIN_PCT",
        "EFG%": "Favorite_EFG",
        "EFG%D": "Favorite_EFGD",
        "TOV%": "Favorite_TOV",
        "TOV%D": "Favorite_TOVD",
        "OREB%": "Favorite_OREB",
        "DREB%": "Favorite_DREB",
        "2PT%": "Favorite_2PT",
        "2PT%D": "Favorite_2PTD",
        "3PT%": "Favorite_3PT",
        "3PT%D": "Favorite_3PTD",
        "AST%": "Favorite_AST",
        "OP OREB%": "Favorite_OP_OREB",
        "OP DREB%": "Favorite_OP_DREB",
        "RAW T": "Favorite_RAW_T",
    })

    dog = team_features.rename(columns={
        "TEAM NO": "UnderdogTeamNo",
        "TEAM": "UnderdogTeamNameFromStats",
        "SEED": "UnderdogSeedFromStats",
        "KADJ EM": "Underdog_KADJ_EM",
        "KADJ O": "Underdog_KADJ_O",
        "KADJ D": "Underdog_KADJ_D",
        "BADJ EM": "Underdog_BADJ_EM",
        "BADJ O": "Underdog_BADJ_O",
        "BADJ D": "Underdog_BADJ_D",
        "BARTHAG": "Underdog_BARTHAG",
        "GAMES": "Underdog_GAMES",
        "W": "Underdog_W",
        "L": "Underdog_L",
        "WIN%": "Underdog_WIN_PCT",
        "EFG%": "Underdog_EFG",
        "EFG%D": "Underdog_EFGD",
        "TOV%": "Underdog_TOV",
        "TOV%D": "Underdog_TOVD",
        "OREB%": "Underdog_OREB",
        "DREB%": "Underdog_DREB",
        "2PT%": "Underdog_2PT",
        "2PT%D": "Underdog_2PTD",
        "3PT%": "Underdog_3PT",
        "3PT%D": "Underdog_3PTD",
        "AST%": "Underdog_AST",
        "OP OREB%": "Underdog_OP_OREB",
        "OP DREB%": "Underdog_OP_DREB",
        "RAW T": "Underdog_RAW_T",
    })

    merged = games_df.merge(
        fav,
        on=["YEAR", "FavoriteTeamNo"],
        how="left"
    )

    merged = merged.merge(
        dog,
        on=["YEAR", "UnderdogTeamNo"],
        how="left"
    )

    return merged


# --------------------------------------------------
# 6. CREATE DIFFERENCE FEATURES
# --------------------------------------------------
def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Differences are underdog minus favorite.
    Positive values mean the underdog looks stronger on that metric.
    """
    df = df.copy()

    diff_pairs = [
        ("KADJ_EM_Diff", "Underdog_KADJ_EM", "Favorite_KADJ_EM"),
        ("KADJ_O_Diff", "Underdog_KADJ_O", "Favorite_KADJ_O"),
        ("KADJ_D_Diff", "Underdog_KADJ_D", "Favorite_KADJ_D"),
        ("BADJ_EM_Diff", "Underdog_BADJ_EM", "Favorite_BADJ_EM"),
        ("BARTHAG_Diff", "Underdog_BARTHAG", "Favorite_BARTHAG"),
        ("WIN_PCT_Diff", "Underdog_WIN_PCT", "Favorite_WIN_PCT"),
        ("EFG_Diff", "Underdog_EFG", "Favorite_EFG"),
        ("EFGD_Diff", "Underdog_EFGD", "Favorite_EFGD"),
        ("TOV_Diff", "Underdog_TOV", "Favorite_TOV"),
        ("TOVD_Diff", "Underdog_TOVD", "Favorite_TOVD"),
        ("OREB_Diff", "Underdog_OREB", "Favorite_OREB"),
        ("DREB_Diff", "Underdog_DREB", "Favorite_DREB"),
        ("TWO_PT_Diff", "Underdog_2PT", "Favorite_2PT"),
        ("THREE_PT_Diff", "Underdog_3PT", "Favorite_3PT"),
        ("AST_Diff", "Underdog_AST", "Favorite_AST"),
    ]

    for new_col, dog_col, fav_col in diff_pairs:
        if dog_col in df.columns and fav_col in df.columns:
            df[new_col] = df[dog_col] - df[fav_col]

    return df


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

    # still print to terminal
    for line in output:
        print(line)

    return "\n".join(output)

# --------------------------------------------------
# 8. MODEL
# --------------------------------------------------
def build_model(df: pd.DataFrame):
    model_df = df.copy()

    feature_cols = [
        "SeedDiff",
        "KADJ_EM_Diff",
        "KADJ_O_Diff",
        "KADJ_D_Diff",
        "BADJ_EM_Diff",
        "BARTHAG_Diff",
        "WIN_PCT_Diff",
        "EFG_Diff",
        "EFGD_Diff",
        "TOV_Diff",
        "TOVD_Diff",
        "OREB_Diff",
        "DREB_Diff",
        "TWO_PT_Diff",
        "THREE_PT_Diff",
        "AST_Diff",
    ]

    feature_cols = [c for c in feature_cols if c in model_df.columns]

    X = model_df[feature_cols]
    y = model_df["Upset"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(cm)

    print("\nROC-AUC:")
    print(auc)

    coefs = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": coefs
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

    games_df = build_games_from_matchups(matchups)
    team_features = prep_team_features(kenpom)
    merged = merge_team_features(games_df, team_features)
    merged = add_difference_features(merged)

    # --------------------------------------------------
    # ADD CONFERENCE INFO
    # --------------------------------------------------
    team_with_conf = kenpom[["TEAM", "CONF", "YEAR"]]

    merged = merged.merge(
        team_with_conf,
        left_on=["YEAR", "UnderdogTeam"],
        right_on=["YEAR", "TEAM"],
        how="left"
    )

    merged = merged.rename(columns={"CONF": "UnderdogConf"})
    merged = merged.drop(columns=["TEAM"])

    print("\nGames preview:")
    print(games_df.head())

    print("\nMerged preview:")
    print(merged.head())

    eda_text = basic_upset_analysis(merged)

    model, coef_df, y_test, preds, probs, model_text = build_model(merged)

    create_pdf_report(eda_text, model_text)

    return merged, model, coef_df


if __name__ == "__main__":
    merged_data, model, coefficients = main()
