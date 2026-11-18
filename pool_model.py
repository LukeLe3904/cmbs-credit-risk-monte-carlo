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

# vectorized simulation
# run_pool_simulation above loops over every path and every loan and is kept as the reference implementation
# the functions below compute the same results with numpy arrays (one axis for paths, one for loans, one for months)

# step 1 of the vectorized simulation: build every loan's mortgage schedule into arrays with one row per loan 
# and one column per month (shape n_loans x term_months)

def build_schedule_arrays(loan_tape):
    n_loans = len(loan_tape)
    term_months = int(loan_tape["Term Years"].max()) * 12

    payment = np.zeros((n_loans, term_months))
    beginning_balance = np.zeros((n_loans, term_months))
    balloon = np.zeros(n_loans)

    for i in range(n_loans):
        loan = loan_tape.iloc[i]
        schedule = create_loan_schedule(
            loan_balance = loan["Loan Balance"],
            annual_rate = loan["Annual Rate"],
            term_years = int(loan["Term Years"]),
            amortization_years = int(loan["Amortization Years"]),
            io_months = int(loan["IO Months"])
        )
        # row i holds loan i's monthly payment and beginning balance for months 1 to term_months
        payment[i, :] = schedule["Payment"].to_numpy()
        beginning_balance[i, :] = schedule["Beginning Balance"].to_numpy()
        # the balloon is the balance left after the final scheduled payment
        balloon[i] = schedule.iloc[-1]["Balloon Payment"]

    return payment, beginning_balance, balloon

# step 2a: draw the shocks for every path at once (shape n_paths x (1 + n_loans))
# column 0 is the systematic shock and columns 1 to n_loans are the idiosyncratic shocks
# each row uses the same random numbers, in the same order, as one call to draw_shocks, so with the same seed
# the shocks are identical to the loop version
def draw_all_shocks(n_paths, betas, rng):
    shocks = rng.standard_normal((n_paths, 1 + len(betas)))
    systematic_shock = shocks[:, 0]
    idiosyncratic_shock = shocks[:, 1:]
    # systematic_shock[:, None] turns the shock into a column so it is applied to every loan on its path,
    # and betas is applied to every path (composite has shape n_paths x n_loans)
    composite = betas * systematic_shock[:, None] + np.sqrt(1 - betas**2) * idiosyncratic_shock
    return systematic_shock, composite

# step 2b: translate the shocks into stress for every path and loan at once, with the same rules as the loop version
# noi_decline has shape n_paths x n_loans and cap_rate_shift has shape n_paths
def calculate_stress(systematic_shock, composite, sigma_noi, cap_sensitivity, max_decline):
    # a one-sigma bad draw is a sigma_noi decline, a good draw is no decline, and the decline is capped at max_decline
    noi_decline = np.clip(sigma_noi * (-composite), 0.0, max_decline)
    # cap rates only widen in a downturn (negative systematic shock)
    cap_rate_shift = np.where(systematic_shock < 0, cap_sensitivity * (-systematic_shock), 0.0)
    return noi_decline, cap_rate_shift

# step 3: build the monthly NOI, cap rate and property value for every path, loan and month at once
# (shape n_paths x n_loans x term_months), following the same rules as create_scenario_path with no NOI growth:
# NOI is flat at base until shock_month, then declines geometrically to the trough over decline_months and stays flat,
# and the cap rate jumps to the shocked cap rate after shock_month
def build_value_arrays(loan_tape, noi_decline, cap_rate_shift, shock_month, decline_months):
    term_months = int(loan_tape["Term Years"].max()) * 12
    base_noi = (loan_tape["Loan Balance"] * loan_tape["Debt Yield"]).to_numpy()
    base_cap_rate = loan_tape["Base Cap Rate"].to_numpy()

    # months 1 to term_months, and how many months past the shock each month is (0 before the shock)
    months = np.arange(1, term_months + 1)
    months_since_shock = np.clip(months - shock_month, 0, None)
    # share of the decline reached by each month: 0 before the shock, rising to 1 after decline_months and staying at 1
    decline_progress = np.minimum(months_since_shock, decline_months) / decline_months

    # NOI_t = base NOI * (1 - noi_decline) ** decline_progress_t, the closed form of the loop's month-by-month decline
    # [:, :, None] adds the month axis to the (paths x loans) declines, [None, :, None] adds the path and month axes to the base NOI
    noi = base_noi[None, :, None] * (1 - noi_decline[:, :, None]) ** decline_progress[None, None, :]

    # cap rate is the base cap rate up to shock_month and base + shift after
    after_shock = months > shock_month
    cap_rate = base_cap_rate[None, :, None] + cap_rate_shift[:, None, None] * after_shock[None, None, :]

    property_value = noi / cap_rate

    return noi, cap_rate, property_value

# step 4: find the term default for every path and loan at once, with the same double trigger as run_loan_scenario:
# a loan defaults in the first month where DSCR < default_dscr and LTV > default_ltv (LTV on the beginning balance)
# returns is_term_default (True/False), default_index (month index 0 to term_months - 1, only meaningful where is_term_default),
# and the balance and property value at default (shape n_paths x n_loans for all four)
def find_term_defaults(payment, beginning_balance, noi, property_value, default_dscr = 1.0, default_ltv = 0.95):
    # payment and beginning_balance are (n_loans x term_months), so [None, :, :] adds the path axis
    dscr = (noi / 12) / payment[None, :, :]
    ltv = beginning_balance[None, :, :] / property_value

    # True in every month where both triggers fail (shape n_paths x n_loans x term_months)
    trigger = (dscr < default_dscr) & (ltv > default_ltv)

    # a loan term defaults if the trigger is hit in any month
    is_term_default = trigger.any(axis = 2)
    # argmax returns the first index of the largest value along the month axis, and True (1) is larger than False (0),
    # so it gives the first trigger month (it returns 0 when there is no trigger, which is why is_term_default is needed)
    default_index = trigger.argmax(axis = 2)

    # pick each path and loan's balance and value in its default month
    n_paths, n_loans = default_index.shape
    loan_index = np.arange(n_loans)
    balance_at_default = beginning_balance[loan_index[None, :], default_index]
    path_index = np.arange(n_paths)
    value_at_default = property_value[path_index[:, None], loan_index[None, :], default_index]

    return is_term_default, default_index, balance_at_default, value_at_default

# step 5a: refinance test at maturity for every path and loan at once, same as run_refinance_test with amortizing sizing
# balloon has shape n_loans, final_noi and final_cap_rate have shape n_paths x n_loans
# returns True where the loan can refinance (shape n_paths x n_loans)
def can_refinance_all(balloon, final_noi, final_cap_rate, refinance_rate, refinance_amortization_years, min_dscr, max_ltv, min_debt_yield):
    # the annuity payment on 1 dollar times 12 gives the cost per dollar of loan amount each year
    monthly_rate = refinance_rate / 12
    n_months = refinance_amortization_years * 12
    annual_constant = 12 * monthly_rate * (1 + monthly_rate)**n_months / ((1 + monthly_rate)**n_months - 1)

    property_value = final_noi / final_cap_rate

    # maximum proceeds a lender is willing to lend based on DSCR, LTV and DY constraints
    dscr_limit = final_noi / (min_dscr * annual_constant)
    ltv_limit = max_ltv * property_value
    debt_yield_limit = final_noi / min_debt_yield
    # the lender lends the smallest of the three limits, np.minimum takes the smaller value element by element
    max_proceeds = np.minimum(np.minimum(dscr_limit, ltv_limit), debt_yield_limit)

    # balloon[None, :] adds the path axis so each loan's balloon is compared on every path
    return max_proceeds >= balloon[None, :]

# step 5b: loss severity for every path and loan at once, same as calculate_loss_severity
# balance_at_default and value_at_default have shape n_paths x n_loans, annual_rate has shape n_loans
def calculate_loss_all(balance_at_default, value_at_default, annual_rate, months_to_resolution, liquidation_discount, liquidation_costs, special_servicing_fee):
    sale_price = value_at_default * (1 - liquidation_discount)
    net_recovery = sale_price * (1 - liquidation_costs)
    accrued_interest = balance_at_default * annual_rate[None, :] / 12 * months_to_resolution
    servicing_fees = balance_at_default * special_servicing_fee / 12 * months_to_resolution
    total_exposure = balance_at_default + accrued_interest + servicing_fees
    # np.maximum replaces max() so the floor at 0 is applied element by element
    return np.maximum(total_exposure - net_recovery, 0.0)

# step 5c: assign every path and loan its outcome and loss, following the same order as run_loan_scenario:
# term default first, then the refinance test, then the maturity LTV gate on the balloon balance
def assign_outcomes(loan_tape, balloon, noi, cap_rate, property_value, is_term_default, term_balance, term_value, refinance_assumptions, workout_assumptions, default_ltv = 0.95):
    # the refinance test and maturity gate use the last month's NOI, cap rate and value (shape n_paths x n_loans)
    final_noi = noi[:, :, -1]
    final_cap_rate = cap_rate[:, :, -1]
    final_value = property_value[:, :, -1]

    can_refinance = can_refinance_all(balloon, final_noi, final_cap_rate, **refinance_assumptions)
    # the borrower has no equity left at maturity if the balloon is more than default_ltv of the property value
    no_equity = balloon[None, :] / final_value > default_ltv

    # each outcome is True only where the earlier outcomes did not happen, so exactly one is True for every path and loan
    is_performed = ~is_term_default & can_refinance
    is_maturity_default = ~is_term_default & ~can_refinance & no_equity
    is_cured = ~is_term_default & ~can_refinance & ~no_equity

    # outcome labels, filled in one outcome at a time
    outcome = np.empty(is_term_default.shape, dtype = object)
    outcome[is_term_default] = "Term Default"
    outcome[is_performed] = "Performed"
    outcome[is_maturity_default] = "Maturity Default"
    outcome[is_cured] = "Refinance Shortfall, Cured"

    # balance and value at default: the term default month's numbers for term defaults, the balloon and final value for maturity defaults
    balance_at_default = np.where(is_term_default, term_balance, balloon[None, :])
    value_at_default = np.where(is_term_default, term_value, final_value)

    # calculate the loss for every path and loan, then keep it only where the loan defaulted
    annual_rate = loan_tape["Annual Rate"].to_numpy()
    loss_if_default = calculate_loss_all(balance_at_default, value_at_default, annual_rate, **workout_assumptions)
    is_default = is_term_default | is_maturity_default
    loss = np.where(is_default, loss_if_default, 0.0)

    return outcome, loss

# step 6: vectorized version of run_pool_simulation, same arguments and same two DataFrames returned
def run_pool_simulation_vectorized(loan_tape, n_paths, refinance_assumptions, workout_assumptions, sigma_noi = 0.15, cap_sensitivity = 0.01, shock_month = 48, decline_months = 24, max_decline = 0.6, seed = 1, batch_size = 2000):
    rng = np.random.default_rng(seed)
    n_loans = len(loan_tape)
    total_balance = loan_tape["Loan Balance"].sum()
    betas = loan_tape["Beta"].to_numpy()
    loan_ids = loan_tape["Loan ID"].to_numpy().astype(int)

    # step 1: mortgage schedules for all loans, built once
    payment, beginning_balance, balloon = build_schedule_arrays(loan_tape)

    path_tables = []
    loan_tables = []

    for batch_start in range(0, n_paths, batch_size):
        batch_end = min(batch_start + batch_size, n_paths)
        n_batch = batch_end - batch_start

        # step 2: shocks and stress for every path in the batch
        systematic_shock, composite = draw_all_shocks(n_batch, betas, rng)
        noi_decline, cap_rate_shift = calculate_stress(systematic_shock, composite, sigma_noi, cap_sensitivity, max_decline)

        # step 3: monthly NOI, cap rate and property value
        noi, cap_rate, property_value = build_value_arrays(loan_tape, noi_decline, cap_rate_shift, shock_month, decline_months)

        # step 4: term defaults
        is_term_default, default_index, term_balance, term_value = find_term_defaults(payment, beginning_balance, noi, property_value)

        # step 5: outcome and loss for every path and loan
        outcome, loss = assign_outcomes(loan_tape, balloon, noi, cap_rate, property_value, is_term_default, term_balance, term_value, refinance_assumptions, workout_assumptions)

        # pool results for each path: sum the loan losses and count the defaults across the loan axis
        path_numbers = np.arange(batch_start, batch_end)
        pool_loss = loss.sum(axis = 1)
        n_default = ((outcome == "Term Default") | (outcome == "Maturity Default")).sum(axis = 1)

        path_table = {}
        path_table["Path"] = path_numbers
        path_table["Systematic Shock"] = systematic_shock
        path_table["Pool Loss"] = pool_loss
        path_table["Loss Rate"] = pool_loss / total_balance
        path_table["Default Rate"] = n_default / n_loans
        path_tables.append(pd.DataFrame(path_table))

        # loan results: flatten the (paths x loans) arrays into one row per path and loan, in the same order as the loop
        # np.repeat gives each path number once per loan, np.tile repeats the list of loan IDs once per path
        loan_table = {}
        loan_table["Path"] = np.repeat(path_numbers, n_loans)
        loan_table["Loan ID"] = np.tile(loan_ids, n_batch)
        loan_table["Systematic Shock"] = np.repeat(systematic_shock, n_loans)
        loan_table["NOI Decline"] = noi_decline.ravel()
        loan_table["Outcome"] = outcome.ravel()
        loan_table["Loss"] = loss.ravel()
        loan_tables.append(pd.DataFrame(loan_table))

    path_results = pd.concat(path_tables, ignore_index = True)
    loan_results = pd.concat(loan_tables, ignore_index = True)

    return path_results, loan_results


# closed-form Vasicek loss quantile at confidence alpha for an infinitely granular pool with default probability pd_ (trailing "_" to not be confused with pd as in pandas)
def vasicek_loss_quantile(pd_, lgd, rho, alpha):
    return lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))