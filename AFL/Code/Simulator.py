import pandas as pd
import numpy as np
from collections import defaultdict
import random


# Config 
SALARY         = 100000
POS_REQ        = {"FWD": 2, "MID": 4, "DEF": 2, "RK": 1}
JITTER_MU      = 0.05     
JITTER_SD      = 0.1
JITTER_OWN     = 0.1
BETA_BASE      = 10            # salary-gap penalty strength
BETA_MIN       = 3.0
SEED           = 741
TRY_LIMIT      = 100


# Simulates scores over n scenarios
def sim_players(store, scenarios = 300):
    """
    args:
    - df: slate store object.
    - scenarios: number of scenarios (default 300).

    Returns:
    scores_arr: np array of S*n_p player scores

    """

    rng = np.random.default_rng(SEED)
    mu = store['proj']
    sigma = store['sd']
    n_p = len(store["pid"])

    scores_arr = np.empty((scenarios, n_p), dtype=np.float32)

    for s in range(scenarios):
        mu_prime    = mu + rng.normal(0, JITTER_MU * mu)
        sigma_prime = sigma * rng.lognormal(0, JITTER_SD, size=n_p)
        scores_arr[s]   = rng.normal(mu_prime, sigma_prime)

    return scores_arr

### Simulates contest field of lineups
def sim_field(store, field_size, scenarios = 300):
    """
    args:
    - store: a dict of numpy arrays.
    - field_size: number of lineups in the contest.
    - scenarios: number of simualted scenarios (default 300).
    """

    rng    = np.random.default_rng(SEED)

    field  = []

    for s in range(scenarios):
        field_s = []
        while len(field_s) < field_size:
            lu = build_field_lineup(store, rng)
            if lu:
                field_s.append(lu)
        field.append(field_s)
    return field



### Builds a single sample lineup
def build_field_lineup(store, rng):
    """
    args:
    - store: a dict of numpy arrays.
    - rng: numpy random generator object
    """

    nP        = len(store["pid"])       
    price     = store['price']
    used_mask = np.zeros(nP, dtype=bool)
    own = store['own']
    # Jitter ownership
    own_prime = own * rng.lognormal(0, JITTER_OWN, size=nP)

    # Max fails
    for attempt in range(TRY_LIMIT):
        need        = POS_REQ.copy()
        salary_left = SALARY
        lu          = []
        lu_pos = []
        while need:
            
            # need dict split
            need_keys = list(need.keys())
            need_values = list(need.values())

            # Randomly pick from remaining positions
            pos = random.choices(need_keys, weights=need_values, k=1)[0]

            # Player pool
            pool_idx = store['pos_idx'][pos] # Position indices
        
            pool_mask = ( 
                (price[pool_idx] <= salary_left) &
                (~used_mask[pool_idx])
            ) # Mask for other rules
 
            pool = pool_idx[pool_mask] # Indices of eligible players
            if pool.size == 0:
                return None # retry

            own_slice   = own_prime[pool]        # projected ownership
            price_slice = price[pool]      # prices
            slots_left = sum(need.values())         # roster slots still empty

            # Calculate the ownership penalty
            weight_penalty = find_penalty(price_slice, salary_left, slots_left)
            
            weights = own_slice * weight_penalty / 100    
            total_weight = weights.sum()
            if total_weight == 0:             
                return None  # retry
            
            # Choose a player
            if len(pool) == 1: # one player eligible
                pick = pool[0]
            else:
                pick = rng.choice(pool, p = weights / total_weight)

            lu.append(pick)
            lu_pos.append(pos)
            used_mask[pick] = True
            salary_left -= price[pick]


            need[pos]    -= 1    
            if need[pos] == 0: # remove filled positions
                del need[pos]

        if len(lu) != sum(POS_REQ.values()):
            continue

        # Fixes low spend lineups
        if salary_left > 10000: # Far off, just try again
            continue

        if salary_left > 5000:
            low_idx = min(lu, key=lambda i: own[i]) # Find the lowest owned guy

            lu_idx = lu.index(low_idx) # Swappee index

            low_pos = lu_pos[lu_idx] # Pos
            low_price = price[low_idx] # Price

            max_budget = salary_left + low_price # Available salary
            min_budget = max_budget - 5000

            # Swap pool
            swap_pool_idx = store['pos_idx'][low_pos] # Position indices

            swap_pool_mask = ( 
                (price[swap_pool_idx] <= max_budget) &
                (price[swap_pool_idx] >= min_budget) &
                (~used_mask[swap_pool_idx])
            ) # Mask for other rules

            swap_pool = swap_pool_idx[swap_pool_mask]
            if swap_pool.size == 0:
                continue # No eligible swaps, just rebuild

            swap_own_slice = own_prime[swap_pool]  
            total_pool_weight = swap_own_slice.sum()       

            # Choose a player
            if len(swap_pool) == 1: # one player eligible
                swap_pick = swap_pool[0]
            else:
                swap_pick = rng.choice(swap_pool, p = swap_own_slice / total_pool_weight)

            # Swap player
            lu[lu_idx] = swap_pick
            used_mask[low_idx] = False
            used_mask[swap_pick] = True

        return lu

### Calculates the payoff for simulated lineups
def build_scenarios(scores, fields):
    """
    args:
    - scores: numpy array of scores.
    - fields: simulated public fields.
    """

    S = len(scores) # Number of scenarios
    scenario_cache = []

    for s in range(S): # Loop through every scenario

        field_pts_s = np.array(
            [scores[s, lu].sum() for lu in fields[s]],
            dtype=np.float32
        )

        sorted_pts_s = np.sort(field_pts_s)[::-1]

        uniq, counts = np.unique(sorted_pts_s, return_counts=True) # 

        # Number of line-ups with score > uniq[i]
        higher_before = np.cumsum(np.insert(counts, 0, 0))[:-1]

        # Map score -> (higher, dup_count)
        meta_s = dict(zip(uniq, zip(higher_before, counts)))

        scenario_cache.append((sorted_pts_s, meta_s))

    return scenario_cache



def find_penalty(prices, salary_left, slots_left):
    """
    args:
    - prices: array of player prices.
    - salary_left: salary left for current lineup
    - slots_left: slots left to fill in lineup
    """

    avg_salary = salary_left / slots_left
    beta_eff = BETA_MIN + (BETA_BASE - BETA_MIN) * (1 - slots_left / 9)       
    dev_ratio = (prices - avg_salary) / avg_salary
    return np.exp(-beta_eff * dev_ratio**2)           


# Parses the df into NumPy Arrays
def build_store(df):
    """
    args:
    - df: slate DataFrame.
    """

    pid      = df["Player_ID"].to_numpy()
    pid_idx  = {p: i for i, p in enumerate(pid)}  # Dict with pid as key, index as value
    price    = df["Price"].to_numpy()
    proj     = df['Projection'].to_numpy()
    sd       = df['SD'].to_numpy()
    own      = df["Projected_Ownership"].to_numpy().astype(float)

    # position masks (indices)
    pos_idx = {pos: np.flatnonzero((df["Position1"] == pos) | (df["Position2"] == pos))
        for pos in ["FWD", "MID", "DEF", "RK"]
    }

    # store arrays in a dict for quick access
    store = dict(pid=pid, pid_idx = pid_idx, price=price, own=own, pos_idx=pos_idx, proj=proj, sd=sd)

    return store

if __name__ == "__main__":
    id = 230520251
    file = f"DFS/AFL/Slates/{id}.csv" 
    df = pd.read_csv(file)

    store = build_store(df) 

    scores = sim_players(store, 10)

    field = sim_field(store, 100, 10)

    scenario_cache = build_scenarios(scores, field) # 