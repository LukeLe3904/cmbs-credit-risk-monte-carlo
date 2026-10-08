from loan_model import (
    create_loan_schedule,
    create_property_credit_summary,
    run_property_stress_scenario,
    get_balloon_balance,
    run_refinance_test,
    calculate_loss_severity,
    get_balance_at_month,
    create_scenario_path,
    run_loan_scenario
)
from validation import validate_loan_schedule
from pool_model import create_sample_loan_tape, run_pool_simulation, vasicek_loss_quantile
import pandas as pd
import matplotlib.pyplot as plt


# inputs for a single loan characteristics
# amortization_years is length of the full amortization schedule, i.e. loan is paid off fully after amortization_years
loan_balance = 10_000_000
annual_rate = 0.06
term_years = 10
amortization_years = 30
io_months = 24

df = create_loan_schedule(loan_balance=loan_balance, annual_rate=annual_rate, term_years=term_years, amortization_years=amortization_years, io_months=io_months)

# inputs for property performance 
revenue = 1_500_000
operating_expense = 600_000
property_value = 15_000_000

# create a credit summary of the property
# year is the year used to calculate annual debt servie (default at 3 to capture the amortization payments instead of the io payments which is first two years)
credit_summary = create_property_credit_summary(
    revenue=revenue,
    operating_expense=operating_expense,
    property_value=property_value,
    mortgage_schedule=df,
    year=3
)

print(credit_summary)

# inputs for stress assumptions
noi_decline = 0.2
stressed_cap_rate = 0.075

# create a credit summary of the property under a stress scenario
stress_summary = run_property_stress_scenario(
    noi=credit_summary['NOI'],
    annual_debt_service=credit_summary['Annual Debt Service'],
    loan_balance=credit_summary['Loan Balance'],
    noi_decline=noi_decline,
    stressed_cap_rate=stressed_cap_rate
)

print(stress_summary)

# inputs for refinance assumptions
refinance_rate = 0.08
refinance_amortization_years = 30
min_dscr = 1.25
max_ltv = 0.65
min_debt_yield = 0.09

# inputs for capitalization rate
cap_rate = 0.06
stressed_cap_rate = 0.075

refinance_assumptions = {
    "refinance_rate": refinance_rate,
    "refinance_amortization_years": refinance_amortization_years,
    "min_dscr": min_dscr,
    "max_ltv": max_ltv,
    "min_debt_yield": min_debt_yield
}

# refinance test at maturity: base case and stress case, sized on the amortizing payment
refinance_base = run_refinance_test(
    balloon_balance=get_balloon_balance(df),
    noi=credit_summary['NOI'],
    cap_rate=cap_rate,
    **refinance_assumptions
)
print(refinance_base)

refinance_stress = run_refinance_test(
    balloon_balance=get_balloon_balance(df),
    noi=stress_summary['Stressed NOI'],
    cap_rate=stressed_cap_rate,
    **refinance_assumptions
)
print(refinance_stress)

# inputs for workout assumptions
months_to_resolution=24
liquidation_discount=0.15 
liquidation_costs=0.08
special_servicing_fee=0.0025

workout_assumptions = {
    "months_to_resolution": months_to_resolution,
    "liquidation_discount": liquidation_discount, 
    "liquidation_costs": liquidation_costs,
    "special_servicing_fee": special_servicing_fee
}

# create a loss summary for a maturity default
loss_summary = calculate_loss_severity(
    loan_balance_at_default=get_balloon_balance(df), 
    property_value=stress_summary['Stressed Value'], 
    annual_rate=annual_rate, 
    **workout_assumptions
)

print(loss_summary)

# base case: NOI and property value remains the same
base_case_path = create_scenario_path(
    base_noi=credit_summary['NOI'], 
    base_cap_rate=cap_rate,
    term_years=term_years    
)

base_case_monthly, base_case_result = run_loan_scenario(
    mortgage_schedule=df, 
    scenario_path=base_case_path,
    annual_rate=annual_rate,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)

print(base_case_result)

# stress case: shock at month 48. 25% over 24 months, 7.5% cap
stress_case_path = create_scenario_path(
    base_noi=credit_summary['NOI'],
    base_cap_rate=cap_rate,
    term_years=term_years,
    shock_month=48,
    noi_decline=0.25,
    decline_months=24,
    shocked_cap_rate=0.075
) 

stress_case_monthly, stress_case_result = run_loan_scenario(
    mortgage_schedule=df,
    scenario_path=stress_case_path,
    annual_rate=annual_rate,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)

print(stress_case_result)

# gradual decline case: NOI declines gradually at 3%
gradual_decline_case_path = create_scenario_path(
    base_noi=credit_summary['NOI'],
    base_cap_rate=cap_rate,
    term_years=term_years,
    noi_growth=-0.03
) 

gradual_decline_case_monthly, gradual_decline_case_result = run_loan_scenario(
    mortgage_schedule=df,
    scenario_path=gradual_decline_case_path,
    annual_rate=annual_rate,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)

print(gradual_decline_case_result)

# build a 30-loan sample pool
loan_tape = create_sample_loan_tape(n_loans=30, seed=0)
print(loan_tape.head())
print("Total Pool Balance:", loan_tape["Loan Balance"].sum())

# base case: 200 simulated futures of the pool under the default model parameters
path_results, loan_results = run_pool_simulation(
    loan_tape=loan_tape,
    n_paths=200,
    refinance_assumptions=refinance_assumptions,
    workout_assumptions=workout_assumptions
)

print("Mean Loss rate: ", round(path_results["Loss Rate"].mean(), 4)) 
print("95th Percentile Loss Rate: ", round(path_results["Loss Rate"].quantile(0.95), 4))
print("99th Percentile Loss Rate:", round(path_results["Loss Rate"].quantile(0.99), 4))
print("Max Loss Rate:", round(path_results["Loss Rate"].max(), 4))
print("Mean Default Rate:", round(path_results["Default Rate"].mean(), 4))
print(loan_results["Outcome"].value_counts())

# correlation sensitivity: same pool, same shocks, every loan's beta forced to one value
# shows that correlation moves the tail of the loss distribution far more than the mean
beta_rows = []
for beta in [0.0, 0.3, 0.6, 0.9]:
    beta_tape = loan_tape.copy()
    beta_tape["Beta"] = beta
    beta_paths, beta_loans = run_pool_simulation(
        loan_tape = beta_tape,
        n_paths = 200,
        refinance_assumptions = refinance_assumptions,
        workout_assumptions = workout_assumptions
    )
    beta_rows.append({
        "Beta": beta,
        "Mean Loss Rate": beta_paths["Loss Rate"].mean(),
        "95th Percentile Loss Rate": beta_paths["Loss Rate"].quantile(0.95),
        "99th Percentile Loss Rate": beta_paths["Loss Rate"].quantile(0.99),
        "Mean Default Rate": beta_paths["Default Rate"].mean()
    })
beta_table = pd.DataFrame(beta_rows)
print(beta_table.round(4).to_string(index = False))
 
# compare the simulated tail against the closed-form Vasicek quantile using the base case's own PD, LGD and average rho
# LGD = total loss on defaulted loans / total balance of defaulted loans
balance_lookup = {}
for i in range(len(loan_tape)):
    loan = loan_tape.iloc[i]
    balance_lookup[int(loan["Loan ID"])] = loan["Loan Balance"]
 
defaulted_balance = 0.0
defaulted_loss = 0.0
for i in range(len(loan_results)):
    row = loan_results.iloc[i]
    if row["Outcome"] == "Term Default" or row["Outcome"] == "Maturity Default":
        defaulted_balance += balance_lookup[int(row["Loan ID"])]
        defaulted_loss += row["Loss"]
 
pool_pd = path_results["Default Rate"].mean()
pool_lgd = defaulted_loss / defaulted_balance
pool_rho = (loan_tape["Beta"] ** 2).mean()
 
print("PD:", round(pool_pd, 4), "LGD:", round(pool_lgd, 4), "rho:", round(pool_rho, 4))
print("Vasicek expected loss:", round(pool_pd * pool_lgd, 4))
print("Simulated mean loss rate:", round(path_results["Loss Rate"].mean(), 4))
print("Vasicek 99th pct loss rate:", round(vasicek_loss_quantile(pd_ = pool_pd, lgd = pool_lgd, rho = pool_rho, alpha = 0.99), 4))
print("Simulated 99th pct loss rate:", round(path_results["Loss Rate"].quantile(0.99), 4))

# NOI volatility sensitivity: 
# note 0.15 is the base case
sigma_rows = []
for sigma_noi in [0.10, 0.15, 0.20]:
    sigma_paths, sigma_loans = run_pool_simulation(
        loan_tape = loan_tape,
        n_paths = 200,
        refinance_assumptions = refinance_assumptions,
        workout_assumptions = workout_assumptions,
        sigma_noi = sigma_noi
    )
    sigma_rows.append({
        "Sigma NOI": sigma_noi,
        "Mean Loss Rate": sigma_paths["Loss Rate"].mean(),
        "95th Percentile Loss Rate": sigma_paths["Loss Rate"].quantile(0.95),
        "99th Percentile Loss Rate": sigma_paths["Loss Rate"].quantile(0.99),
        "Mean Default Rate": sigma_paths["Default Rate"].mean()
    })
sigma_table = pd.DataFrame(sigma_rows)
print(sigma_table.round(4).to_string(index = False))

# cap rate sensitivity
# note 0.01 is the base case
cap_rows = []
for cap_sensitivity in [0.0, 0.01, 0.02]:
    cap_paths, cap_loans = run_pool_simulation(
        loan_tape = loan_tape,
        n_paths = 200,
        refinance_assumptions = refinance_assumptions,
        workout_assumptions = workout_assumptions,
        cap_sensitivity = cap_sensitivity
    )
    cap_rows.append({
        "Cap Sensitivity": cap_sensitivity,
        "Mean Loss Rate": cap_paths["Loss Rate"].mean(),
        "95th Percentile Loss Rate": cap_paths["Loss Rate"].quantile(0.95),
        "99th Percentile Loss Rate": cap_paths["Loss Rate"].quantile(0.99),
        "Mean Default Rate": cap_paths["Default Rate"].mean()
    })
cap_table = pd.DataFrame(cap_rows)
print(cap_table.round(4).to_string(index = False))

# histogram of the base case path loss rates with the mean, 95th and 99th percentile marked
# saved as a PNG for the README
mean_loss_rate = path_results["Loss Rate"].mean()
percentile_95_loss_rate = path_results["Loss Rate"].quantile(0.95)
percentile_99_loss_rate = path_results["Loss Rate"].quantile(0.99)

plt.figure(figsize = (9, 5))
plt.hist(path_results["Loss Rate"], bins = 30, color = "steelblue", edgecolor = "white")
plt.axvline(mean_loss_rate, color = "black", linestyle = "--", label = "Mean = {:.1%}".format(mean_loss_rate))
plt.axvline(percentile_95_loss_rate, color = "orange", linestyle = "--", label = "95th percentile = {:.1%}".format(percentile_95_loss_rate))
plt.axvline(percentile_99_loss_rate, color = "red", linestyle = "--", label = "99th percentile = {:.1%}".format(percentile_99_loss_rate))
plt.xlabel("Pool loss rate")
plt.ylabel("Number of paths")
plt.title("Simulated pool loss distribution (30 loans, 200 paths)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_distribution.png", dpi = 150)
plt.close()
print("Saved loss_distribution.png")