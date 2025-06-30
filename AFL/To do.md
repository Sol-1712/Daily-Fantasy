TO DO:

- Fix Speed and Uniqueness -> Warm start
    - I think do manaul exposures for now, then later backtest and automate ownership penalties etc
- Add in Exposures - Automatic?
- Add stacking
    - Scaling, Mid/Fwd, Mid/RK, (KeyDef/KeyFwd), (Tag/Tagged), Player Groups

    
    - For scaling, do:

    Compute each team’s implied total:
    𝑇𝐴 = (190.5+8.5)/2 = 99.5, 𝑇𝐵 = (190.5−8.5)/2 = 91.0.

    Compute the sum of your players’ baseline projections on each team:
    ∑p∈A μp​ = SA​, ∑𝑝∈𝐵 𝜇𝑝 = 𝑆𝐵​.

    Scale each player 𝑝 on Team A by:
    μp′​ = μp × SA/TA
​    Repeat for Team B
 
​


- Upgrade objective function (expected value) - Includes Covariance



- Cash games - Not sure as they are sort of dead for AFL
- Better ownership
- Own Projections







CASH:
Simulate player scores
Generate expected lineups using ownership % and projections - Jiggle 
Find the lineup with the highest chance of scoring above the (45th) percentile cutoff
Enter it a bunch


For each scenario:
Simulate the lineup field (jitter ownership),  M lineups
Draw player scores from dist.
Score each M lineups
FInd the cutoff