### TO DO: 
### COMBINE DATA INTO ONE CSV - NEED TO BE ABLE TO ADD EACH NEW SLATE
### HAVE THE GENERAL 

import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_SET = "DFS/NBA_dataset.csv"

### Compiles and adds slate to main dataset
def build_dataset(proj, scores):
    data = pd.read_csv(DATA_SET)

    # Formatting
    proj.rename(columns={"ID": "Player_ID", "Date": "Slate_ID"}, inplace = True)
    proj['Player_ID'] = proj['Player_ID'].astype(int)
    proj = proj[proj['Status'] != 'OUT OF GAME']
    scores.rename(columns={'Date': 'Slate_ID', 'Player ID': 'Player_ID'}, inplace = True)
    scores = scores[['Slate_ID', 'Player_ID', 'FPPG','Form','Playing Status','Score','Selected %','Exclude']]
    scores = scores.drop_duplicates(subset = ['Player_ID'])

    # Merge Proj and Score
    proj = proj.merge(scores, on=['Slate_ID', 'Player_ID'], how = 'left')

    # Removing empty rows
    proj = proj.dropna(subset=["Selected %"])
    proj = proj[['Slate_ID'] + [col for col in proj.columns if col != 'Slate_ID']]

    # Featuere building
    proj = add_features(proj)

    # Add merge to full dataset
    merged = pd.concat([data, proj], ignore_index=True)
    merged.to_csv("DFS/NBA_dataset.csv", index = False)

### Adds the features for ownership model
def add_features(data):
    features = ['Player_ID', 'Opponent', 'Status', 'Price', 'Projection', 'SD', 'Ceiling', 'PG', 'SG', 'SF', 'PF', 'C', 'Dual_Position', 'Viable_Player',
                'Value', 'Pos1_Value_Rank', 'Pos2_Value_Rank', 'Avg_PG_Value', 'Avg_SG_Value', 'Avg_SF_Value', 'Avg_PF_Value', 'Avg_C_Value',
                'Viable_PG_Count', 'Viable_SG_Count', 'Viable_SF_Count', 'Viable_PF_Count', 'Viable_C_Count']
    
    # Categorising Features
    encoder = LabelEncoder()
    data["Opponent"] = encoder.fit_transform(data["Opponent"])

    for pos in ["PG", "SG", "SF", "PF", "C"]:
        data[pos] = (data["Position"] == pos) | (data["Position2"] == pos)
    data[["PG", "SG", "SF", "PF", "C"]] = data[["PG", "SG", "SF", "PF", "C"]].astype(int)

    data["Status"] = data["Status"].fillna("BENCH")
    data["Status"] = data["Status"].replace("ESTIMATED STARTER", "STARTER")
    PreGame_Map = {
        'STARTER': 0,
        'BENCH': 1,
        'GAME TIME DECISION': 3
    }
    data["Status"] = data["Status"].map(PreGame_Map)

    data['Dual_Position'] = data['Position2'].notna().astype(int)

    # Viable player if projected >= 12 (Can change)
    data["Viable_Player"] = data["Projection"] >= 12
    data["Viable_Player"] = data["Viable_Player"].astype(int)

    # Value Features
    data['Value'] = data['Projection'] / data['Price'] * 1000

    data["Pos_Value_Rank"] = data.groupby(["Slate_ID", "Position"])["Value"].rank(ascending=False)
    data["Pos2_Value_Rank"] = data.groupby(["Slate_ID", "Position2"])["Value"].rank(ascending=False)

    # Viable and Avg features
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        data[f"Viable_{pos}_Count"] = (
            data[(data[pos] == 1) & (data["Viable_Player"])]  
            .groupby("Slate_ID")["Player_ID"] 
            .transform("count") 
        )
        avg_pos_value = (
            data[data[pos] == 1]
            .groupby("Slate_ID")["Value"]
            .mean()
        )
        data[f'Avg_{pos}_Value'] = data["Slate_ID"].map(avg_pos_value)

    return data

if __name__ == '__main__':
    date = input("Enter Game Date: ")
    proj = pd.read_csv(f"DFS/Projections/{date}-Proj.csv")
    scores = pd.read_csv(f"DFS/Scores/{date}-Score.csv")
    build_dataset(proj, scores)


