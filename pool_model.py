import numpy as np
import pandas as pd
from scipy.stats import norm
from loan_model import create_loan_schedule, create_scenario_path, run_loan_scenario

# create a random pools of CMBS loans with predefined ranges for loan characteristics
def create_sample_loan_tape(n_loans, seed = 0):
    rng = np.random.default_rng(seed)

    loans = []

    for i in range(1, n_loans+1):
        loan_id = i
        loan_balance = round(rng.uniform(5_000_000, 40_000_000), -5)
        annual_rate = round(rng.uniform(0.05, 0.07), 4)
        term_years = 10
        amortization_years = 30
        io_months = int(rng.choice([0, 24, 60, 120]))
        debt_yield = round(rng.uniform(0.085, 0.11), 4)
        base_cap_rate = round(rng.uniform(0.05, 0.07), 4)
        # beta represents the correlation of the loan with the market
        beta = round(rng.uniform(0.4, 0.8), 2)
        loans.append({
            "Loan ID": loan_id,
            "Loan Balance": loan_balance,
            "Annual Rate": annual_rate,
            "Term Years": term_years,
            "Amortization Years": amortization_years,
            "IO Months": io_months,
            "Debt Yield": debt_yield,
            "Base Cap Rate": base_cap_rate,
            "Beta": beta
        })

    return  pd.DataFrame(loans)

# draw one market factor shared by every borrower and one idiosyncratic factor per borrower and combine them so each borrower's composite shock has unit variance 
# and correlation beta__i with the market.
# correlation between borrower i and borrower j is beta_i*beta_j
# derives from the Vasicek Asymptotic Single-Risk-Factor (ASRF) model: composite = beta_i * systematic_shock + sqrt(1 - beta_i^2) * idiosyncratic_shock_i
    # systematic_shock is the systematic shock and reflects the state of the overall economy (one draw for every loan) ~ N(0,1) ==> CRE market performance in related CMBS sector
    # idiosyncratic_shock_i is the borrower i's idiosyncratic factor (one draw for each loan) ~ N(0,1) ==> borrower specific risk factors
    # composite is the borrower i's standardized asset-performance shock and Var(Z_i) = 1
def draw_shocks(n_loans, betas, rng):
    systematic_shock = rng.standard_normal()
    idiosyncratic_shock = rng.standard_normal(n_loans)
    composite = betas * systematic_shock + np.sqrt(1 - betas**2) * idiosyncratic_shock
    return systematic_shock, composite

# simulate the loan pool over n_paths draws of the single-factor model and return the pool loss on each path and the outcome of every loan on every path
# sigma_noi economically represents the volatility of NOI over the shock horizon
def run_pool_simulation(loan_tape, n_paths, refinance_assumptions, workout_assumptions, sigma_noi = 0.15, cap_sensitivity = 0.01, shock_month = 48, decline_months = 24, max_decline = 0.6, seed = 1):
    rng = np.random.default_rng(seed)
    n_loans = len(loan_tape)
    total_balance = loan_tape["Loan Balance"].sum()
    betas = loan_tape["Beta"].to_numpy()

    # build mortgage schedules for all loans before running simulation
    schedules = {}
    for i in range(n_loans):
        loan = loan_tape.iloc[i]
        schedules[loan["Loan ID"]] = create_loan_schedule(
            loan_balance = loan["Loan Balance"],
            annual_rate = loan["Annual Rate"],
            term_years = int(loan["Term Years"]),
            amortization_years = int(loan["Amortization Years"]),
            io_months = int(loan["IO Months"])
        )

    path_results = []
    loan_results = []

    for path in range(n_paths):
        systematic_shock, composite_shock = draw_shocks(n_loans, betas, rng)

        # cap rates respond to the systematic shock only and only widen in a downturn (negative shock)
        if systematic_shock < 0:
            cap_rate_shift = cap_sensitivity * (-systematic_shock)
        else:
            cap_rate_shift = 0.0

        path_loss = 0.0
        n_default = 0

        for i in range(n_loans):
            loan = loan_tape.iloc[i]

            # translate the loan's composite shock into an NOI decline: a one-sigma bad draw is a sigma_noi decline,
            # a good draw is no decline, and the decline is capped at max_decline
            raw_decline = sigma_noi * (-composite_shock[i])
            if raw_decline < 0:
                noi_decline = 0.0
            elif raw_decline > max_decline:
                noi_decline = max_decline
            else:
                noi_decline = raw_decline

            # NOI is implied from the loan's debt yield
            scenario_path = create_scenario_path(
                base_noi = loan["Loan Balance"] * loan["Debt Yield"],
                base_cap_rate = loan["Base Cap Rate"],
                term_years = int(loan["Term Years"]),
                shock_month = shock_month,
                noi_decline = noi_decline,
                decline_months = decline_months,
                shocked_cap_rate = loan["Base Cap Rate"] + cap_rate_shift
            )

            monthly, result = run_loan_scenario(
                mortgage_schedule = schedules[loan["Loan ID"]],
                scenario_path = scenario_path,
                annual_rate = loan["Annual Rate"],
                refinance_assumptions = refinance_assumptions,
                workout_assumptions = workout_assumptions
            )

            if result["Loss Summary"] is None:
                loss = 0.0
            else:
                loss = result["Loss Summary"]["Loss Amount"]

            if result["Outcome"] == "Term Default" or result["Outcome"] == "Maturity Default":
                n_default += 1

            path_loss += loss

            loan_results.append({
                "Path": path,
                "Loan ID": int(loan["Loan ID"]),
                "Systematic Shock": systematic_shock,
                "NOI Decline": noi_decline,
                "Outcome": result["Outcome"],
                "Loss": loss
            })

        path_results.append({
            "Path": path,
            "Systematic Shock": systematic_shock,
            "Pool Loss": path_loss,
            "Loss Rate": path_loss / total_balance,
            "Default Rate": n_default / n_loans
        })

    return pd.DataFrame(path_results), pd.DataFrame(loan_results)

# closed-form Vasicek loss quantile at confidence alpha for an infinitely granular pool with default probability pd_ (trailing "_" to not be confused with pd as in pandas)
def vasicek_loss_quantile(pd_, lgd, rho, alpha):
    return lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))