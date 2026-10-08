import pandas as pd
import numpy_financial as npf

# create a Panda DataFrame for a mortgage schedule given loan balanace, annual rate, loan term, amort period and io period
def create_loan_schedule(loan_balance, annual_rate,  term_years, amortization_years, io_months):

    # check for invalid inputs
    if loan_balance <= 0:
        raise ValueError("Loan balance must be greater than 0")
    if annual_rate < 0:
        raise ValueError("Annual rate cannot be negative")
    if term_years <= 0:
        raise ValueError("Loan term must be greater than 0")
    if amortization_years < 0:
        raise ValueError("Amortization period cannot be negative")
    if io_months < 0 or io_months > term_years*12:
        raise ValueError("Interest-only months must be between 0 and the loan term in months")

    monthly_rate = annual_rate/12
    amortization_months = amortization_years*12
    amortizing_months_in_term = term_years*12 - io_months

    # checks for valid amort and io structure and calculates for the appropriate amortizing payments
    if amortizing_months_in_term > 0 and amortization_months == 0:
        raise ValueError("Amortization period must be positive when the loan has amortizing months")
    if amortizing_months_in_term > amortization_months:
        raise ValueError("Amortizing months in the term cannot exceed the amortization period")

    if amortizing_months_in_term == 0:
        amortizing_payment = 0
    elif monthly_rate == 0:
        amortizing_payment = loan_balance / amortization_months
    else:
        amortizing_payment = loan_balance * monthly_rate / (1 - (1 + monthly_rate)**(-amortization_months))

    beginning_balance = loan_balance
    schedule = []

    # populate the partial io periods
    for month in range(1, io_months + 1):
        io_payment = beginning_balance*monthly_rate
        principal = 0
        ending_balance = beginning_balance - principal
        schedule.append({
                    "Month": month,
                    "Beginning Balance": beginning_balance,
                    "Payment": io_payment,
                    "Interest": io_payment,
                    "Principal": principal,
                    "Ending Balance": ending_balance
                    })
        beginning_balance = ending_balance

    # populate the amort periods
    for month in range(io_months+1, term_years*12+1):

        interest = beginning_balance * monthly_rate
        principal = amortizing_payment - interest
        ending_balance = beginning_balance - principal
        schedule.append({
            "Month": month,
            "Beginning Balance": beginning_balance,
            "Payment": amortizing_payment,
            "Interest": interest,
            "Principal": principal,
            "Ending Balance": ending_balance
            })
        beginning_balance = ending_balance

    df = pd.DataFrame(schedule)

    # set balloon payment to zero for all intermediate months and set ending balance at end of loan term to balloon payment
    df['Balloon Payment'] = 0.0
    balloon_balance = df.iloc[-1]['Ending Balance']
    df.loc[df.index[-1], 'Balloon Payment'] = balloon_balance
    df['Final Ending Balance'] = df['Ending Balance'] - df['Balloon Payment']

    return df

# calculate property NOI given revenue and operating expense
def calculate_noi(revenue, operating_expense):
    noi = revenue - operating_expense
    return noi

# calculate annual debt service for a given year within a mortgae schedule
def calculate_annual_debt_service(mortgage_schedule, year):
    if (year < 1 or year > len(mortgage_schedule) // 12):
        raise ValueError("Invalid year. Year must be between 1 and the loan term")
    start = year*12 - 12
    end = year*12
    return mortgage_schedule.iloc[start:end]['Payment'].sum()

# calculate debt service coverage ratio given NOI and annual debt service
def calculate_dscr(noi, annual_debt_service):
    dscr = noi / annual_debt_service
    return dscr

# calculate debt yield given NOI and loan balance
def calculate_debt_yield(noi, loan_balance):
    debt_yield = noi / loan_balance
    return debt_yield

# calculate loan-to-value ratio given loan balance and property value
def calculate_ltv(loan_balance, property_value):
    ltv = loan_balance / property_value
    return ltv

# return a property credit summary given revenue, operating expense, property value, mortgage schedule
# the input "year" is to obtain the corresponding annual debt service for that given year from the mortgage schedule (default = 1)
def create_property_credit_summary(revenue, operating_expense, property_value, mortgage_schedule, year = 1):
    # calculate property credit metrics using predefined functions
    annual_debt_service = calculate_annual_debt_service(mortgage_schedule, year)
    loan_balance = mortgage_schedule.iloc[(year-1)*12]['Beginning Balance']
    noi = calculate_noi(revenue, operating_expense)
    dscr = calculate_dscr(noi, annual_debt_service)
    debt_yield = calculate_debt_yield(noi, loan_balance)
    ltv = calculate_ltv(loan_balance, property_value)

    credit_summary = {
        "Loan Balance": loan_balance,
        "NOI": noi,
        "Annual Debt Service": annual_debt_service,
        "DSCR": dscr,
        "Debt Yield": debt_yield,
        "LTV": ltv
    }

    return credit_summary

# return property credit summary in a stress scenario given stress on NOI and cap rate
def run_property_stress_scenario(noi, annual_debt_service, loan_balance, noi_decline, stressed_cap_rate):

    stressed_noi = noi*(1-noi_decline)
    stressed_value = stressed_noi / stressed_cap_rate
    stressed_dscr = stressed_noi / annual_debt_service
    stressed_debt_yield = stressed_noi / loan_balance
    stressed_ltv = loan_balance / stressed_value

    stress_summary = {
        "Stressed NOI": stressed_noi,
        "Stressed Value": stressed_value,
        "Stressed DSCR": stressed_dscr,
        "Stressed Debt Yield": stressed_debt_yield,
        "Stressed LTV": stressed_ltv
    }

    return stress_summary

# obtain balloon balance from mortgage schedule
def get_balloon_balance(mortgage_schedule):
    balloon_balance = mortgage_schedule.iloc[-1]['Balloon Payment']
    return balloon_balance

# return property refinance summary and feasibility given constraints on CRE metrics 
def run_refinance_test(balloon_balance, noi, cap_rate, refinance_rate, refinance_amortization_years, min_dscr, max_ltv, min_debt_yield, sizing_basis = 'amortizing'):
    # calculate annual_constant, which represents the cost per one dollar of loan amount each year
    if sizing_basis == "io":
        # if sizing basis is based on io, the refinance_rate is the cost per dollar of loan amount each year
        annual_constant = refinance_rate
    elif sizing_basis == "amortizing":
        # if sizing basis is based on amortizating payment, the annuity payment on a loan balance of 1 dollar times 12 gives the cost per dollar of loan amount each year
        if refinance_amortization_years <= 0:
            raise ValueError("Amortizing sizing requires a positive amortization period")
        monthly_rate = refinance_rate / 12
        n_months = refinance_amortization_years * 12
        annual_constant = 12 * monthly_rate * (1 + monthly_rate)**n_months / ((1 + monthly_rate)**n_months - 1)
    else:
        raise ValueError("Sizing basis must be 'amortizing' or 'io'")

    property_value = noi / cap_rate

    # calculate the maximum proceeds a lender is willing to lend based on DSCR, LTV and DY constraints
    dscr_limit = noi / (min_dscr*annual_constant)
    ltv_limit = max_ltv * property_value
    debt_yield_limit = noi / min_debt_yield
    limits = {
        "DSCR": dscr_limit,
        "LTV": ltv_limit,
        "Debt Yield": debt_yield_limit
    }
    # to satisfy all three constraints, the lender is willing to lend the minimum of those limits
    binding_constraint = min(limits, key = limits.get)
    max_proceeds = limits[binding_constraint]

    refinance_gap = max(balloon_balance - max_proceeds, 0)

    refinance_summary = {
        "Property Value": property_value,
        "DSCR Limit": dscr_limit,
        "LTV Limit": ltv_limit,
        "Debt Yield Limit": debt_yield_limit,
        "Max Proceeds": max_proceeds,
        "Binding Constraint": binding_constraint,
        "Refinancing Gap": refinance_gap,
        "Can Refinance": bool(max_proceeds >= balloon_balance),
        "Sizing Basis": sizing_basis
    }

    return refinance_summary

# calculate the loss severity and return loss summary given default
def calculate_loss_severity(loan_balance_at_default, property_value, annual_rate, months_to_resolution = 24, liquidation_discount = 0.15, liquidation_costs = 0.08, special_servicing_fee = 0.0025):
    sale_price = property_value * (1 - liquidation_discount)
    net_recovery = sale_price * (1 - liquidation_costs)
    accrued_interest = loan_balance_at_default * annual_rate / 12 * months_to_resolution
    servicing_fees = loan_balance_at_default * special_servicing_fee / 12 * months_to_resolution
    total_exposure = loan_balance_at_default + accrued_interest + servicing_fees
    loss_amount = max(total_exposure - net_recovery, 0)
    # standard convention for loss severity quotes by most rating agencies and servicers 
    loss_severity = loss_amount / loan_balance_at_default

    loss_summary = {
            "Sale Price": sale_price,
            "Net Recovery": net_recovery,
            "Accrued Interest": accrued_interest,
            "Servicing Fees": servicing_fees,
            "Total Exposure": total_exposure,
            "Loss Amount": loss_amount,
            "Loss Severity": loss_severity,
    }

    
    return loss_summary

# helper function for loan_balance_at_default: return the balance at term default
def get_balance_at_month(mortgage_schedule, month):
    if month < 1 or month > len(mortgage_schedule):
        raise ValueError("Month must be between 1 and the loan term in months")
    # the balance owed when that month payment is missed
    return mortgage_schedule.iloc[month-1]['Beginning Balance']

# create a simple, deterministic path of property NOI and value
def create_scenario_path(base_noi, base_cap_rate, term_years, noi_growth = 0.0, shock_month = None, noi_decline = 0.0, decline_months = 1, shocked_cap_rate = None):
    term_months = term_years*12
    if shock_month is None:
        shock_month = term_months
    if shocked_cap_rate is None:
        shocked_cap_rate = base_cap_rate

    schedule = []
    
    # populate the schedule where base_noi grows at noi_growth until shock_month then declines to noi_decline * base_noi over decline_months 
    # and stays flat after while base_cap_rate jumps to shocked_cap_rate 
    for month in range(1, shock_month + 1):
        noi = base_noi * (1 + noi_growth / 12)
        property_value = noi / base_cap_rate
        schedule.append({
            "Month": month,
            "NOI": noi,
            "Cap Rate": base_cap_rate,
            "Property Value": property_value
        })
        base_noi = noi

    # noi_decline represents the total peak-to-trough NOI drawdown
    trough_noi = base_noi * (1 - noi_decline)
    monthly_noi_factor = (trough_noi/base_noi) ** (1 / decline_months)

    for month in range(shock_month + 1, term_months + 1):
        # noi declines geometrically to trough_noi then stays flat after decline_months
        noi = max(base_noi * monthly_noi_factor, trough_noi)
        property_value = noi / shocked_cap_rate
        schedule.append({
            "Month": month,
            "NOI": noi,
            "Cap Rate": shocked_cap_rate, 
            "Property Value": property_value
        })
        base_noi = noi

    df = pd.DataFrame(schedule)

    return df

# run the loan scenario given the mortgage schedule and the scenario path 
# default_ltv = 1 - 5% (assumed selling costs)
def run_loan_scenario(mortgage_schedule, scenario_path, annual_rate, refinance_assumptions, workout_assumptions, default_dscr = 1.0, default_ltv = 0.95):
    monthly = mortgage_schedule.merge(scenario_path, on = "Month")
    monthly['DSCR'] = (monthly['NOI'] / 12) / monthly['Payment']
    # LTV is based on beginning balance because to determine the LTV threshold at term default, the beginning balance reflects
    # accurately that the payment for the month when the borrower defaults is missed
    monthly['LTV'] = monthly['Beginning Balance'] / monthly['Property Value']

    # create a boolean column for months when both DSCR and LTV fail threshold, implying the borrower will default
    monthly['Term Default Trigger'] = (monthly['DSCR'] < default_dscr) & (monthly['LTV'] > default_ltv) 

    triggered = monthly[monthly['Term Default Trigger']]

    # if triggered is not empty, there is term default, else the loan reaches maturity meeting all DSCR and LTV thresholds
    if len(triggered) > 0:
        outcome = "Term Default"
        default_month = int(triggered.iloc[0]['Month'])
        balance_at_default = get_balance_at_month(mortgage_schedule, default_month)
        value_at_default = triggered.iloc[0]['Property Value']

    else:
        # run a refinance test to check if the loan can be refinanced
        refinance = run_refinance_test(balloon_balance = get_balloon_balance(mortgage_schedule), noi = monthly.iloc[-1]['NOI'], cap_rate = monthly.iloc[-1]['Cap Rate'], **refinance_assumptions)
        if refinance['Can Refinance']:
            outcome = 'Performed'
            default_month = None
            balance_at_default = None
            value_at_default = None
        else:
            # if selling the property net selling costs at 0.95 of its value is not enough to cover the balloon payment,
            # the borrower has no equity left and can't refinance, leading to a maturity default
            if get_balloon_balance(mortgage_schedule) / monthly.iloc[-1]['Property Value'] > default_ltv:
                outcome = "Maturity Default"
                default_month = int(monthly.iloc[-1]['Month'])
                balance_at_default = get_balloon_balance(mortgage_schedule)
                value_at_default = monthly.iloc[-1]['Property Value']
            # the borrower still has equity left and is willing to put in cash to cover the refinance shortfall to cure the default 
            else: 
                outcome = "Refinance Shortfall, Cured"
                default_month = None
                balance_at_default = None
                value_at_default = None

    if default_month is None:
        loss_summary = None
    else:
        loss_summary = calculate_loss_severity(balance_at_default, value_at_default, annual_rate, **workout_assumptions)

    loan_scenario = {
        "Outcome": outcome,
        "Default Month": default_month,
        "Balance at Default": balance_at_default,
        "Value at Default": value_at_default,
        "Loss Summary": loss_summary
    }

    return monthly, loan_scenario

