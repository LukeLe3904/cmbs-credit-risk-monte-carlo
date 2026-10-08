import numpy as np
import pandas as pd
from scipy.stats import norm
from pool_model import create_sample_loan_tape, draw_shocks, run_pool_simulation, vasicek_loss_quantile
from loan_model import create_loan_schedule

# tolerances: one cent on dollar amounts, floating-point noise on quantities that are the same calculation,
# and a few standard errors on sample correlations over 20,000 draws
money_tolerance = 0.01
float_tolerance = 1e-9
stat_tolerance = 0.03

# inputs for refinance assumptions
refinance_assumptions = {
    "refinance_rate": 0.08,
    "refinance_amortization_years": 30,
    "min_dscr": 1.25,
    "max_ltv": 0.65,
    "min_debt_yield": 0.09
}

# inputs for workout assumptions
workout_assumptions = {
    "months_to_resolution": 24,
    "liquidation_discount": 0.15,
    "liquidation_costs": 0.08,
    "special_servicing_fee": 0.0025
}

# helper: call draw_shocks n_draws times for a given set of betas and stack the results
# returns the market draws as a 1-D array (one per draw) and the composite draws as a 2-D array (rows = draws, columns = loans)
def collect_draws(betas, n_draws, seed):
    rng = np.random.default_rng(seed)
    market_draws = []
    composite_draws = []
    for draw in range(n_draws):
        market, composite = draw_shocks(n_loans = len(betas), betas = betas, rng = rng)
        market_draws.append(market)
        composite_draws.append(composite)
    return np.array(market_draws), np.array(composite_draws)

# tape test: every loan characteristic must fall inside the range it was drawn from,
# loan ids must run 1, 2, ..., n and the seed-0 total balance must match the known check figure
tape = create_sample_loan_tape(n_loans = 30, seed = 0)
assert len(tape) == 30
assert list(tape["Loan ID"]) == list(range(1, 31))
for i in range(len(tape)):
    loan = tape.iloc[i]
    assert 5_000_000 <= loan["Loan Balance"] <= 40_000_000
    assert loan["Loan Balance"] % 100_000 == 0
    assert 0.05 <= loan["Annual Rate"] <= 0.07
    assert loan["Term Years"] == 10
    assert loan["Amortization Years"] == 30
    assert loan["IO Months"] in [0, 24, 60, 120]
    assert 0.085 <= loan["Debt Yield"] <= 0.11
    assert 0.05 <= loan["Base Cap Rate"] <= 0.07
    assert 0.4 <= loan["Beta"] <= 0.8
# total portfolio balance at 717_900_000 corresponds to seed 0 
assert abs(tape["Loan Balance"].sum() - 717_900_000) < money_tolerance

# reproducibility test: the same seed must give the same tape and a different seed must not
assert create_sample_loan_tape(n_loans = 30, seed = 0).equals(tape)
assert not create_sample_loan_tape(n_loans = 30, seed = 1).equals(tape)

# factor test, beta = 1: sqrt(1 - 1^2) = 0 elimninates the idiosyncratic term, so every composite
# shock must equal the market shock exactly
rng = np.random.default_rng(1)
market, composite = draw_shocks(n_loans = 3, betas = np.array([1.0, 1.0, 1.0]), rng = rng)
for i in range(3):
    assert abs(composite[i] - market) < float_tolerance

# factor test, beta = 0: beta * market = 0, so the composite shock is pure idiosyncratic noise.
# over 20,000 draws each loan's variance must be about 1 and its correlation with the market and with the other loans must be about 0
market_draws, composite_draws = collect_draws(betas = np.array([0.0, 0.0, 0.0]), n_draws = 20_000, seed = 2)
for i in range(3):
    # variance of 20,000 draws has a standard error of about 0.01, so 0.05 is 5 standard errors
    assert abs(np.var(composite_draws[:, i]) - 1.0) < 0.05
    assert abs(np.corrcoef(composite_draws[:, i], market_draws)[0, 1]) < stat_tolerance
    for j in range(i + 1, 3):
        assert abs(np.corrcoef(composite_draws[:, i], composite_draws[:, j])[0, 1]) < stat_tolerance

# factor test, beta = 0.6: each loan's variance must still be about 1,
# each loan's correlation with the market must be about beta = 0.6 and each pair of loans must have correlation about beta^2 = rho = 0.36
market_draws, composite_draws = collect_draws(betas = np.array([0.6, 0.6, 0.6]), n_draws = 20_000, seed = 3)
for i in range(3):
    # variance of 20,000 draws has a standard error of about 0.01, so 0.05 is 5 standard errors
    assert abs(np.var(composite_draws[:, i]) - 1.0) < 0.05
    assert abs(np.corrcoef(composite_draws[:, i], market_draws)[0, 1] - 0.6) < stat_tolerance
    for j in range(i + 1, 3):
        assert abs(np.corrcoef(composite_draws[:, i], composite_draws[:, j])[0, 1] - 0.36) < stat_tolerance

# factor test, mixed betas 0.4 and 0.8: with different loadings the pairwise correlation is
# beta_i * beta_j = 0.32, not a single rho, and each loan still has correlation beta_i with the market
market_draws, composite_draws = collect_draws(betas = np.array([0.4, 0.8]), n_draws = 20_000, seed = 4)
assert abs(np.corrcoef(composite_draws[:, 0], market_draws)[0, 1] - 0.4) < stat_tolerance
assert abs(np.corrcoef(composite_draws[:, 1], market_draws)[0, 1] - 0.8) < stat_tolerance
assert abs(np.corrcoef(composite_draws[:, 0], composite_draws[:, 1])[0, 1] - 0.32) < stat_tolerance

# Vasicek quantile tests:
# test 1: with rho = 0 there is no systematic risk, so every quantile collapses to the expected loss lgd * pd
assert abs(vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.0, alpha = 0.99) - 0.35 * 0.02) < float_tolerance
# test 2: with rho > 0 the loss distribution is right skewed so median < expected loss
assert vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.36, alpha = 0.50) < 0.35 * 0.02
# test 3: the quantile rises with alpha
assert vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.36, alpha = 0.99) > vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.36, alpha = 0.95)
# test 4: the quantilt rises with rho
assert vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.64, alpha = 0.99) > vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.36, alpha = 0.99)
# test 5: with pd = 2%, lgd = 35%, rho = 0.36, alpha = 99%, the vasicek loss equation should match the hand-computed 0.0719
assert abs(vasicek_loss_quantile(pd_ = 0.02, lgd = 0.35, rho = 0.36, alpha = 0.99) - 0.0719) < 0.0001

# pool simulation consistency test on a small pool (5 loans x 20 paths): output shapes, the path table
# must be the loan table rolled up (pool loss = sum of loan losses, loss rate = pool loss / total balance,
# default rate = defaulted loans / n_loans), every loan on a path must share that path's systematic shock,
# losses cannot be negative and NOI declines must stay inside [0, max_decline]
small_tape = create_sample_loan_tape(n_loans = 5, seed = 0)
path_results, loan_results = run_pool_simulation(
    loan_tape=small_tape,
    n_paths=20,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)
assert path_results.shape == (20, 5)
assert loan_results.shape == (100, 6)
total_balance = small_tape["Loan Balance"].sum()
for i in range(len(path_results)):
    path_row = path_results.iloc[i]
    loan_rows = loan_results[loan_results["Path"] == path_row["Path"]]
    assert len(loan_rows) == 5
    assert abs(path_row["Pool Loss"] - loan_rows["Loss"].sum()) < money_tolerance
    assert abs(path_row["Loss Rate"] - path_row["Pool Loss"] / total_balance) < float_tolerance
    n_default = 0
    for j in range(len(loan_rows)):
        loan_row = loan_rows.iloc[j]
        assert abs(loan_row["Systematic Shock"] - path_row["Systematic Shock"]) < float_tolerance
        assert loan_row["Loss"] >= 0
        assert 0 <= loan_row["NOI Decline"] <= 0.6
        if loan_row["Outcome"] == "Term Default" or loan_row["Outcome"] == "Maturity Default":
            n_default += 1
    assert abs(path_row["Default Rate"] - n_default / 5) < float_tolerance

# reproducibility test: the same seed must give identical path and loan tables
path_results_again, loan_results_again = run_pool_simulation(
    loan_tape=small_tape,
    n_paths=20,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)
assert path_results.equals(path_results_again)
assert loan_results.equals(loan_results_again)

# blind-spot test, sigma_noi = 0: no shock can move NOI, so every NOI decline must be exactly zero
# and no loan can hit the DSCR trigger, so there must be no term defaults. Every loan on the tape
# has a debt yield above its annual constant (cost per dollar of loan amount each year), so this holds for any tape drawn by create_sample_loan_tape
path_results, loan_results = run_pool_simulation(
    loan_tape=small_tape,
    n_paths=20,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions,
    sigma_noi=0.0
)
assert (loan_results["NOI Decline"] == 0).all()
assert (loan_results["Outcome"] == "Term Default").sum() == 0

# Merton identity test, end to end: a pool of identical loans with beta = 0 and no cap-rate channel reduces
# the simulation to independent draws of the single-loan model, so the term default frequency across loan-paths
# must match the closed form PD = Phi(-d_star / sigma_noi) where d_star is computed from the loan's own schedule
# d_star is the proportional decline in NOI that will trigger a default
# (DSCR leg = 1 - ADS / NOI, LTV leg = 1 - balance * cap / (0.95 * NOI), both at the trough month 72)

sigma_noi = 0.15
merton_rows = []
for i in range(1, 21):
    merton_rows.append({"Loan ID": i, "Loan Balance": 10_000_000, "Annual Rate": 0.06, "Term Years": 10,
        "Amortization Years": 30, "IO Months": 24, "Debt Yield": 0.09, "Base Cap Rate": 0.06, "Beta": 0.0})
merton_tape = pd.DataFrame(merton_rows)
merton_schedule = create_loan_schedule(loan_balance = 10_000_000, annual_rate = 0.06, term_years = 10, amortization_years = 30, io_months = 24)
base_noi = 10_000_000 * 0.09
balance_at_trough = merton_schedule.iloc[71]["Beginning Balance"]
annual_debt_service = 12 * merton_schedule.iloc[71]["Payment"]
d_dscr = 1 - annual_debt_service / base_noi
d_ltv = 1 - balance_at_trough * 0.06 / (0.95 * base_noi)
d_star = max(d_dscr, d_ltv)
merton_paths, merton_loans = run_pool_simulation(loan_tape = merton_tape, n_paths = 250,
    refinance_assumptions = refinance_assumptions, workout_assumptions = workout_assumptions,
    sigma_noi = sigma_noi, cap_sensitivity = 0.0)
simulated_pd = (merton_loans["Outcome"] == "Term Default").mean()
closed_form_pd = norm.cdf(-d_star / sigma_noi)
# closed_form_pd (Phi(-d_star / sigma_noi)) is about 1.3%, and a 1.3% proportion over 5,000 loan-paths has a standard error of about 0.0016, so 0.006 is four standard errors
assert abs(simulated_pd - closed_form_pd) < 0.006

# correlation ordering test: the same tape run with every beta set to 0.0 and then to 0.9 must show
# a fatter loss tail (higher 99th percentile loss rate) when the loans move together.
beta_tape = create_sample_loan_tape(n_loans=8, seed=0)
low_beta_tape = beta_tape.copy()
low_beta_tape["Beta"] = 0.0
high_beta_tape = beta_tape.copy()
high_beta_tape["Beta"] = 0.9
low_beta_paths, low_beta_loans = run_pool_simulation(
    loan_tape=low_beta_tape,
    n_paths=100,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)
high_beta_paths, high_beta_loans = run_pool_simulation(
    loan_tape=high_beta_tape,
    n_paths=100,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)
assert high_beta_paths["Loss Rate"].quantile(0.99) > low_beta_paths["Loss Rate"].quantile(0.99)

print("All 12 tests passed")