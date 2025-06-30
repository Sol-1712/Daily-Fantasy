import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
import os

### Compiles and adds slate to main dataset
def build_training_dataset(id, type):

    # Load me stuff
    p = f"DFS/AFL/Training_Data/GPP/Proj-{type}/{id}.csv"
    s = f"DFS/AFL/Training_Data/GPP/Score-{type}/{id}.csv"
    d = f"DFS/AFL/AFL/Training_Data/GPP_{type}.csv"
    dataset = pd.read_csv(d, index_col = False)
    proj = pd.read_csv(p)
    score = pd.read_csv(s)

    # Formatting
    proj['Slate_ID'] = id
    proj.rename(columns={"ID": "Player_ID", "Position": "Position1"}, inplace = True)

    score.rename(columns={'Player ID': 'Player_ID'}, inplace = True)
    score = score[['Player_ID', 'FPPG','Form','Playing Status','Score','Selected %']]
    score = score.drop_duplicates(subset = ['Player_ID'])

    # Merge proj and score
    merged = proj.merge(score, on=['Player_ID'], how = 'right')
    merged['Player_ID'] = merged['Player_ID'].astype(int)

    merged = add_features(merged)

    if id not in dataset['Slate_ID'].values:
        # Saving to dataset file
        dataset = pd.concat([dataset, merged], ignore_index=True)
        dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]

        # Fix floats
        float_cols = dataset.select_dtypes(include=["float"]).columns
        dataset[float_cols] = dataset[float_cols].round(4)
        
        dataset.to_csv(d, index = False)


### Adds features to the slate
def add_features(slate):

    positions = ["FWD", "MID", "DEF", "RK"]
    
    # Categorising
    Team_Map = {
        'ADE': 1,
        'BRI': 2,
        'CAR': 3,
        'ESS': 4,
        'COL': 5,
        'FRE': 6,
        'GCS': 7,
        'GEE': 8,
        'GWS': 9,
        'HAW': 10,
        'MEL': 11,
        'NME': 12,
        'POR': 13,
        'RIC': 14,
        'STK': 15,
        'SYD': 16,
        'WBD': 17,
        'WCE': 18
    }
    slate['Team_ID'] = slate['Team'].map(Team_Map)
    slate["Opponent_ID"] = slate["Opponent"].map(Team_Map)

    # Positions
    slate['Dual_Position'] = slate['Position2'].notna().astype(int)
    
    for pos in positions:
        slate[f"{pos}_Count"] = ((slate["Position1"] == pos) | (slate["Position2"] == pos)).sum()
        slate[pos] = (slate["Position1"] == pos) | (slate["Position2"] == pos)
    slate[["FWD", "MID", "DEF", "RK"]] = slate[["FWD", "MID", "DEF", "RK"]].astype(int)

    # Map pre-game status
    slate["Status"] = slate["Status"].replace("Start", "In")
    slate["Status"] = slate["Status"].replace("Not Named", "Out")
    PreGame_Map = {
        'In': 0,
        'Out': 1
    }
    slate["Status_ID"] = slate["Status"].map(PreGame_Map)

    # Value features
    slate['Value'] = slate['Projection'] / slate['Price'] * 1000

    slate["Pos_Value_Rank"] = np.nan
    slate["Pos2_Value_Rank"] = np.nan

    for idx, row in slate.iterrows():
        slate_id = row["Slate_ID"]
        slate_group = slate[slate["Slate_ID"] == slate_id]

        # === Position 1 Rank ===
        pos1 = row["Position1"]
        if pd.notnull(pos1) and pos1 in positions:
            mask = (slate_group[pos1] == 1)
            same_pos_players = slate_group[mask]
            rank = same_pos_players["Value"].rank(ascending=False)
            player_value = row["Value"]
            player_rank = rank[same_pos_players["Value"] == player_value].min()
            slate.at[idx, "Pos_Value_Rank"] = player_rank

        # === Position 2 Rank ===
        pos2 = row["Position2"]
        if pd.notnull(pos2) and pos2 in positions:
            mask = (slate_group[pos2] == 1)
            same_pos2_players = slate_group[mask]
            rank2 = same_pos2_players["Value"].rank(ascending=False)
            player_value2 = row["Value"]
            player_rank2 = rank2[same_pos2_players["Value"] == player_value2].min()
            slate.at[idx, "Pos2_Value_Rank"] = player_rank2

    # Averge features
    for pos in positions:

        avg_pos_value = (
            slate[(slate["Position1"] == pos) | (slate["Position2"] == pos)]
            .groupby("Slate_ID")["Value"]
            .mean()
        )
        slate[f'Avg_{pos}_Value'] = slate["Slate_ID"].map(avg_pos_value)

    return slate


### TRAIN DA MODEL
def train_model(type):
    # Load data
    d = f"DFS/AFL/AFL/Training_Data/GPP_{type}.csv"
    df = pd.read_csv(d)

    target = 'Selected %'
    exclude_cols = ["Slate_ID", "Player_ID", 'Name', 'Team', 'Opponent', 'Status', 'Playing Status', 
                    'Score', 'Position1', 'Position2', target]

    X = df.drop(columns=exclude_cols, errors="ignore")
    y = df[target]

    # Model parameters
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, f"ownership_{type}.pkl")

### Predicts the ownership for the slate
def predict_ownership(slate, type):
    model = joblib.load(f"DFS/AFL/Code/ownership_{type}.pkl")
    
    features = ['Price', 'Projection', 'SD', 'Floor', 'Ceiling', 'FPPG', 'Form', 'Team_ID', 'Opponent_ID', 'Dual_Position',
                'FWD_Count', 'FWD', 'MID_Count', 'MID', 'DEF_Count', 'DEF', 'RK_Count', 'RK', 'Status_ID', 'Value', 'Pos_Value_Rank',
                'Pos2_Value_Rank', 'Avg_FWD_Value', 'Avg_MID_Value', 'Avg_DEF_Value', 'Avg_RK_Value']

    X_new = slate[features]

    slate["Predicted_Ownership"] = model.predict(X_new)

    return slate['Predicted_Ownership']


def daily(id, type):

    # Load files
    p = f"DFS/AFL/Slates/{id}.csv"
    l = f"DFS/AFL/Slates/{id}_L.csv"
    proj = pd.read_csv(p)
    lobby = pd.read_csv(l)

    # Formatting
    proj.rename(columns={"ID": "Player_ID", "Position": "Position1"}, inplace = True)
    lobby.rename(columns={"Player ID": "Player_ID"}, inplace = True)
    lobby = lobby.drop_duplicates(subset = ['Player_ID'])
    lobby = lobby[['Player_ID', 'FPPG', 'Form']]

    # Creating the Slate
    slate = proj.merge(lobby, on=['Player_ID'], how = 'left')
    slate['Slate_ID'] = id
    slate = slate[slate["Projection"] != 0]
    slate = add_features(slate)

    # Saving Ownership calculation
    proj['Projected_Ownership'] = predict_ownership(slate, type)

    proj["Projected_Ownership"] = (proj["Projected_Ownership"] / proj["Projected_Ownership"].sum()) * 900
    proj["Projected_Ownership"] = proj['Projected_Ownership'].fillna(0).round().astype(int)
    proj = proj.merge(lobby, on=['Player_ID'], how = 'left')

    # Rearrange
    proj = proj[proj["Projection"] != 0]
    cols = ['Player_ID', 'Name', 'Team', 'Opponent', 'Status', 'Position1', 'Position2',
            'Price', 'Projection', 'SD', 'Floor', 'Ceiling', 'FPPG', 'Form', 'Projected_Ownership']
    
    proj = proj[cols]
    proj.to_csv(p, index=False)


if __name__ == '__main__':

    id = input("Enter Slate: ")
    #type = input("Enter Type: ")
    type = 'Single'

    #slate_folder = f"DFS/AFL/Training_Data/Proj-{type}"
    # for filename in os.listdir(slate_folder):
    #     if filename.endswith(".csv"):
    #         print(filename)
    #         id = filename.replace(".csv", "")
    #         build_training_dataset(id, type)

    #train_model(type)

    daily(id, type)