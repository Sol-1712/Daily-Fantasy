import pandas as pd
import numpy as np
import Simulator
import os
import sys
import math
import time
from ortools.sat.python import cp_model
import scipy.optimize as opt

CONTEST = 'A'       # Set current contest structure

LINEUP_SIZE=9
SALARY = 100000
MIN_SPEND = SALARY * 0.93
POS_REQ = {"FWD": 2, "MID": 4, "DEF": 2, "RK": 1}

# Objective coefficients
KAPPA = 0.45        # Ceiling scale
LMDA = 1.3          # Chalk Penalty
GAMMA = 1.2           #
ALPHA = 0.2         # Overlap penalty
EPS_RC = 0.01       # Reduced cost floor
OBJ_SCALE = 1000    # Integer Scale

# Contest objects:

CONTEST_A = {
    'ENTRY_FEE': 2,
    'ENTRIES': 15,
    'SIZE': 1744,
    'CUTOFF': 436,
    'LADDER': [
        0.00,        # unused 0
        300.18,      # 1st
        150.10,      # 2nd
         69.96,      # 3rd
         45.95,      # 4th
         39.95,      # 5th
         38.14,      # 6th
         36.03,      # 7th
         33.93,      # 8th
         32.13,      # 9th
         30.03,      # 10th
        *[15.92]*10,     # 11th-20th  (ten identical values)
        *[15.02]*10,     # 21st-30th
        *[12.01]*10,     # 31st-40th
        *[10.01]*10,     # 41st-50th
        *[ 6.00]*50,     # 51st-100th
        *[ 5.00]*50,     # 101st-150th
        *[ 4.00]*286     # 151st-436th   (= 436-150)
    ]

    }

CONTEST_B = {
    'ENTRY_FEE': 2,
    'ENTRIES': 10,
    'SIZE': 872,
    'CUTOFF': 218,
    'LADDER': [
        0.00,
        175.12,
        87.03,
        55.97,
        40.07,
        30.01,
        25.96,
        24.01,
        22.06,
        19.96,
        18.01,
        *[9.01]*10,
        *[7.96]*10,
        *[6.01]*10,
        *[5.00]*60,
        *[4.00]*118
    ]
    }


### Helper function to build lineup mask arrays
def lineup_mask(lineup_idx, n_p):
    """
    args:
    - lineup_idx: lineup index array.
    - n_p: number of players on the slate.
    """

    m = np.zeros(n_p, dtype=np.uint8)  
    m[lineup_idx] = 1                   
    return m                          

### Builds my seed lineups
def build_seed(store, n_seeds=100):
    """
    args:
    - store: parsed slate df object.
    - n_seeds: number of seed lineups.
    - kappa: upside/ceiling scaling.
    - lmda: chalk penalty.
    - alpha: exposure penalty.
    """

    n_p = len(store["pid"])
    used  = np.zeros(n_p, dtype=int)          # exposure counts
    masks = []                               
    lineups  = []                         

    while (len(lineups) < n_seeds):
        model  = cp_model.CpModel()
        x      = [model.NewBoolVar(f"x{i}") for i in range(n_p)]

        # Position requirement
        for pos, need in POS_REQ.items():
            pos_idx = store['pos_idx'][pos]
            model.Add(sum(x[i] for i in pos_idx) == need)

        # Salary and player constraint
        model.Add(sum(store['price'][i] * x[i] for i in range(n_p)) <= SALARY)
        model.Add(sum(store["price"][i] * x[i] for i in range(n_p)) >= int(MIN_SPEND))
        model.Add(sum(x) == LINEUP_SIZE)

        # Hamming >=2
        for old_mask in masks:
            model.Add(sum((1 - old_mask[i]) * x[i] +
                          old_mask[i] * (1 - x[i]) for i in range(n_p)) >= 2)

        # Objective Function
        coeff = (store["proj"]
                 + KAPPA * store["sd"]
                 - LMDA  * store["own"]
                 - ALPHA * used)
        
        model.Maximize(sum(int(coeff[i] * OBJ_SCALE) * x[i] for i in range(n_p)))        


        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            print('uh')
            break   # timed out or infeasible → stop seeding

        # Results
        idx_sel = [i for i in range(n_p) if solver.Value(x[i])]
        mask    = lineup_mask(idx_sel, n_p)

        # skip duplicates (rare with Hamming≥2 but safe)
        if any((mask == m).all() for m in masks):
            continue

        masks.append(mask)
        lineups.append(tuple(store["pid"][idx_sel]))
        used[idx_sel] += 1        

    return masks, lineups


### Finds payoff for a lineup
def lineup_payoff(mask, scenario_cache, scores):

    if CONTEST == 'A':
        contest = CONTEST_A
    else:
        contest = CONTEST_B


    mask = mask.astype(bool)
    idx_list = np.nonzero(mask)[0]

    S = scores.shape[0] # number of scenarios
    profit_vec = np.empty(S, dtype=np.float32) # holds profit for each scenario
    for s, (sorted_pts, meta) in enumerate(scenario_cache):
        pts = scores[s, mask].sum() # lineup points

        if pts in meta: # score ties a public lineup
            higher, dup_pub = meta[pts]
            rank = higher + 1
            dup  = dup_pub + 1     # add to tie count


        else: # unqiue score
            higher = np.searchsorted(-sorted_pts, -pts, side='left')
            rank   = higher + 1
            dup    = 1            # solo
        
        # Prize
        if rank <= contest['CUTOFF']:
            upper = min(rank + dup - 1, contest['CUTOFF'])
            pool  = sum(contest['LADDER'][rank : upper + 1])   # sum inclusive slice
            prize = pool / dup
        else:
            prize = 0.0

        profit_vec[s] = prize - contest['ENTRY_FEE']

    return profit_vec


### Master LP Solver
def master_lp(payoff_matrix):

    if CONTEST == 'A':
        contest = CONTEST_A
    else:
        contest = CONTEST_B

    P = payoff_matrix
    N, S = P.shape
    M = contest['ENTRIES']

    # cost vector
    c = np.hstack([np.zeros(N), -np.ones(S)]).astype(np.float64)      

    # equality
    A_eq = np.zeros((1, N + S))
    A_eq[0, :N] = 1
    b_eq = [M]

    # Upper bounds
    A_ub = np.zeros((S, N + S))
    A_ub[:, :N] = -P.T               #  shape (S, N)
    A_ub[:, N:] = np.eye(S)         #  -z_ω  (negative identity)
    b_ub = np.zeros(S)               # right-hand side = 0

    # Variable bounds
    bounds = [(0, 1)] * N + [(None, None)] * S

    # Solve
    res = opt.linprog(c,
                    A_ub=A_ub, b_ub=b_ub,
                    A_eq=A_eq, b_eq=b_eq,
                    bounds=bounds,
                    method="highs")

    if res.status != 0:             # 0 = optimal
        raise RuntimeError(f"LP failed: {res.message}")

    # Results
    y_frac = res.x[:N]              # optimal fractional mix  (length N)
    dual_z = -res.ineqlin.marginals # dual prices
    
    return y_frac, dual_z


def build_pricing_column(store, masks, y_frac, dual_z, scores, scenario_cache):

    n_p = len(store['pid'])

    # Overlap penalty gamma
    if masks:
        exposure = (np.vstack(masks).T @ y_frac)
    else:
        exposure = np.zeros(n_p)
    overlap_penalty = GAMMA * exposure

    # MILP model
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n_p)]

    for pos, need in POS_REQ.items():
        idx = store["pos_idx"][pos]
        model.Add(sum(x[i] for i in idx) == need)

    model.Add(sum(store['price'][i] * x[i] for i in range(n_p)) <= SALARY)
    model.Add(sum(store["price"][i] * x[i] for i in range(n_p)) >= int(MIN_SPEND))
    model.Add(sum(x) == LINEUP_SIZE)

    # Objective Function
    coeff = (store["proj"]
            + KAPPA * store["sd"]
            - LMDA  * store["own"]
            - overlap_penalty)

    model.Maximize(sum(int(coeff[i] * OBJ_SCALE) * x[i] for i in range(n_p)))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5
    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        return None, None
    
    idx_sel  = [i for i in range(n_p) if solver.Value(x[i])]
    mask_new = np.zeros(n_p, dtype=bool)
    mask_new[idx_sel] = True

    payoff = lineup_payoff(mask_new, scenario_cache, scores)    

    # Reduced cost
    rc = float(np.dot(dual_z, payoff)) - GAMMA * (mask_new @ exposure)
    print(rc)
    if rc <= EPS_RC:
        return None, None # Not good enough lineup

    return mask_new, payoff

if __name__ == "__main__":

    #id = input("Enter slate ID: ")
    id = 230520251
    file = f"DFS/AFL/Slates/{id}.csv" 
    df = pd.read_csv(file)

    store = Simulator.build_store(df)
    scores = Simulator.sim_players(store, 10)
    field = Simulator.sim_field(store, 1744, 10)
    scenario_cache = Simulator.build_scenarios(scores, field)
    
    masks, seed_lus = build_seed(store)

    # Builds payoff matrix
    payoff = np.vstack([lineup_payoff(m, scenario_cache, scores)
                    for m in masks])   
    print(payoff)

    y_frac, dual_z = master_lp(payoff)

    while True:
        print('Here')
        mask_new, payoff_new = build_pricing_column(store, masks, y_frac, dual_z, scores, scenario_cache)

        if mask_new is None: # No new column
            break

        masks.append(mask_new)
        payoff = np.vstack([payoff, payoff_new])
        seed_lus.append(tuple(store['pid'][mask_new]))
        y_frac, dual_z = master_lp(payoff) # Resolve


    print(y_frac)
    print(dual_z)