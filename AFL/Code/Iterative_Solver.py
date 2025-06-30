import pandas as pd
import Ownership
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpBinary, PulpSolverError, LpStatus
import os

LINEUP_SIZE=9
SALARY = 100000




def iterative_lineups(id, type, num_lineups=10,  mode = 'gpp', kappa = 1):
    """
    Args:
    - id: slate ID.
    - type: slate type.
    - num_lineups: number of lineups to generate.
    - mode: cash or gpp contest (default gpp).
    - kappa: Upside emphasis for projections - Increase for single game slates.
    """
    # Read the slate CSV
    proj_path =f"DFS/AFL/Slates/{id}.csv" 
    lineup_path = f"DFS/AFL/Slates/{id}_lineups.csv" 
    slate = pd.read_csv(proj_path)

    # Mapping for easier access
    player_ids = slate['Player_ID'].tolist()
    prices = dict(zip(player_ids, slate['Price']))
    projections = dict(zip(player_ids, slate['Projection']))
    sds = dict(zip(player_ids, slate['SD']))
    min_exposures = dict(zip(player_ids, slate['MIN']))
    max_exposures = dict(zip(player_ids, slate['MAX']))
    positions1 = dict(zip(player_ids, slate['Position1']))
    positions2 = dict(zip(player_ids, slate['Position2']))

    # Used for SD solving
    gpp_score = {
        pid: projections[pid] + kappa * sds[pid]
        for pid in player_ids
    }

    # Positional constraints
    pos_constraints = {'FWD': 2, 'MID': 4, 'DEF': 2, 'RK': 1}  # Fixed number of players for each position

    # Lineups and Exposure objects
    exposure_counts = {pid: 0 for pid in player_ids}
    lineups = []
    lineup_data = []

    # Generate lineups
    for i in range(num_lineups):
        prob = LpProblem(f"Lineup_{i+1}", LpMaximize)
        
        # Binary vars for slotting each player into exactly one position
        x_fwd = {pid: LpVariable(f"x_fwd_{pid}", 0, 1, LpBinary) for pid in player_ids}
        x_mid = {pid: LpVariable(f"x_mid_{pid}", 0, 1, LpBinary) for pid in player_ids}
        x_def = {pid: LpVariable(f"x_def_{pid}", 0, 1, LpBinary) for pid in player_ids}
        x_rk = {pid: LpVariable(f"x_rk_{pid}", 0, 1, LpBinary) for pid in player_ids}

        # Sum all position assignments
        prob += lpSum(x_fwd.values()) == pos_constraints['FWD']
        prob += lpSum(x_mid.values()) == pos_constraints['MID']
        prob += lpSum(x_def.values()) == pos_constraints['DEF']
        prob += lpSum(x_rk.values()) == pos_constraints['RK']

        # One position max per player
        for pid in player_ids:
            prob += x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid] <= 1

            # Respect eligibility
            if 'FWD' not in (positions1[pid], positions2[pid]):
                x_fwd[pid].upBound = 0
            if 'MID' not in (positions1[pid], positions2[pid]):
                x_mid[pid].upBound = 0
            if 'DEF' not in (positions1[pid], positions2[pid]):
                x_def[pid].upBound = 0
            if 'RK' not in (positions1[pid], positions2[pid]):
                x_rk[pid].upBound = 0

        # Salary cap
        prob += lpSum(prices[pid] * (x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid]) for pid in player_ids) <= SALARY
        prob += lpSum(prices[pid] * (x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid]) for pid in player_ids) >= SALARY * 0.95
        
        # Exposure check
        for pid in player_ids:
            max_count = int(num_lineups * (max_exposures[pid] / 100))
            min_count = int(num_lineups * (min_exposures[pid] / 100))
            if exposure_counts[pid] >= max_count:
                prob += x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid] == 0
            elif num_lineups - i <= min_count - exposure_counts[pid]:
                prob += x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid] == 1

        # Unique lineups
        for lineup in lineups:
            prob += lpSum(x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid] for pid in lineup) <= LINEUP_SIZE - 1

        ### ADDITIONAL CONSTRAINTS


        ### SOLVE SECTION

        # Objective: maximize projection 
        if mode == 'cash':
            prob += lpSum(
                projections[pid] * (x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid])
                for pid in player_ids
            )

        # Objective: Maximise proj and SD
        if mode == 'gpp':
            prob += lpSum(
                gpp_score[pid] * (x_fwd[pid] + x_mid[pid] + x_def[pid] + x_rk[pid])
                for pid in player_ids
            )


        # Solve
        try:
            prob.solve()
        except PulpSolverError as e:
            print(e)
            break

        # Check if still feasible
        status = LpStatus[prob.status]
        if status != 'Optimal':
            print(f"Stopping: lineup {i + 1} is not feasible ({status}).")
            break 


        # Parse lineups
        selected_fwds = [pid for pid, var in x_fwd.items() if var.varValue == 1]
        selected_mids = [pid for pid, var in x_mid.items() if var.varValue == 1]
        selected_defs = [pid for pid, var in x_def.items() if var.varValue == 1]
        selected_rks  = [pid for pid, var in x_rk.items()  if var.varValue == 1]

        selected = selected_fwds + selected_mids + selected_defs + selected_rks
        selected_names = []

        # Update exposure count, add names for readability
        for pid in selected:
            name = slate.loc[slate['Player_ID'] == pid, 'Name'].values[0]
            selected_names.append(name)
            exposure_counts[pid] += 1

        final = selected + selected_names
        # Collect lineup
        lineups.append(selected)
        lineup_data.append(final)


        
    #df = pd.DataFrame(lineups, columns = ['FWD1', 'FWD2', 'MID1', 'MID2', 'MID3', 'MID4', 'DEF1', 'DEF2', 'RK1'])
    df = pd.DataFrame(lineup_data, columns = ['FWD1', 'FWD2', 'MID1', 'MID2', 'MID3', 'MID4', 'DEF1', 'DEF2', 'RK1',
                                'F1', 'F2', 'M1', 'M2', 'M3', 'M4', 'D1', 'D2', 'R1'])
    df.to_csv(lineup_path, index = False)

    # Add GPP score to slate
    slate['GPP_Score'] = slate['Player_ID'].map(gpp_score)

    # Update exposures back into slate CSV (as percentage)
    slate['Exposure'] = slate['Player_ID'].map(lambda pid: (exposure_counts[pid] / num_lineups) * 100)
    slate = slate.sort_values(by = 'Exposure', ascending = False)
    slate.to_csv(proj_path, index=False)

    # Opens Slate
    try:
        os.startfile(f"C:/Users/SBBus/Documents/NBA MODEL/{proj_path}")
        print("Iterative Solve Complete!")
    except Exception as e:
        print(f"Error opening file: {e}")



if __name__ == '__main__':
    #id = input('Enter a slate: ')
    #type = input('Enter slate type: ')
    id = 240420251
    type = 'Single'
    iterative_lineups(id, type = type, num_lineups = 20, kappa=1.5)
