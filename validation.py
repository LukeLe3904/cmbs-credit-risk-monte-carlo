import pandas as pd
import numpy_financial as npf

# tolerance: one cent on dollar amounts, since every check in this validator compares dollar figures
money_tolerance = 0.01

# validate a mortgage schedule by testing for common checks given a tolerance of 0.01
def validate_loan_schedule(mortgage_schedule, loan_balance, annual_rate, term_years, amortization_years, io_months, tolerance = money_tolerance):

    findings = []

    required_columns = {
        "Month",
        "Beginning Balance",
        "Payment",
        "Interest",
        "Principal",
        "Ending Balance",
        "Balloon Payment",
        "Final Ending Balance"
    }

    # ensure the required columns are within the mortgage schedule
    if not required_columns.issubset(set(mortgage_schedule.columns)):
        return False, ["Required columns are missing"]

    # ensure there are no principal payments during io periods
    if mortgage_schedule.iloc[:io_months]["Principal"].abs().gt(tolerance).any():
        findings.append("Principal payment does not equal zero for the interest only periods")

    # ensure payment equals interest plus principal
    if (mortgage_schedule["Payment"] - mortgage_schedule["Interest"] - mortgage_schedule["Principal"]).abs().gt(tolerance).any():
        findings.append("Payment does not equal interest + principal")

    # ensure the balance rolls forward correctly
    gap = mortgage_schedule["Beginning Balance"] - mortgage_schedule["Ending Balance"].shift(1)
    # skips the first row because shift(1) to the first month's ending balance results in NaN for the first row as there are no prior month
    if gap.iloc[1:].abs().gt(tolerance).any():
        findings.append("Balance does not roll forward appropriately")

    # ensure ending balance is correctly reduced by principal payment from beginning balance
    if (mortgage_schedule["Ending Balance"] - mortgage_schedule["Beginning Balance"] + mortgage_schedule["Principal"]).abs().gt(tolerance).any():
        findings.append("Ending balance does not equal beginning balance - principal")

    # ensure balloon payment is zero throughout the loan term except for the last month
    if (mortgage_schedule.iloc[:-1]['Balloon Payment']).abs().gt(tolerance).any():
        findings.append("Balloon payment is nonzero for the intermediate months")

    # ensure balloon payment is equal to the ending balance for the last month
    if abs(mortgage_schedule.iloc[-1]["Ending Balance"] - mortgage_schedule.iloc[-1]["Balloon Payment"]) > tolerance:
        findings.append("Balloon payment does not equal ending balance for the last month")

    # ensure the 'final' ending balance is correctly reduced by the balloon payment from the ending balance for the last month 
    if abs(mortgage_schedule.iloc[-1]["Final Ending Balance"] - mortgage_schedule.iloc[-1]["Ending Balance"] + mortgage_schedule.iloc[-1]["Balloon Payment"]) > tolerance:
        findings.append("Final ending balance does not equal ending balance - balloon payment for the last month")

    # ensure the 'final' ending balance is zero
    if abs(mortgage_schedule.iloc[-1]["Final Ending Balance"]) > tolerance:
        findings.append("Final ending balance does not equal 0")

    monthly_rate = annual_rate / 12                 
    amortization_months = amortization_years * 12
    # calculate the number of months within the loan term with amortization          
    amortizing_months_in_term = term_years * 12 - io_months     
    
    # ensure interest is correctly calculated as the product of the beginning balance and the monthly rate
    if (mortgage_schedule["Interest"] - mortgage_schedule["Beginning Balance"]*monthly_rate).abs().gt(tolerance).any():
        findings.append("Monthly interest does not equal beginning balance * monthly rate") 

    # ensure amortizing payment is correctly calculated when benchmarked against numpy pmt function
    # ensure balloon payment is correctly calculated when benchmarked against numpy fv function
    # ensure balloon payment is equal to the closed-form derivation of balloon payment
    if amortizing_months_in_term > 0 and amortization_months > 0:
        # calculate amortizing payment using numpy pmt function as benchmark
        expected_payment = npf.pmt(monthly_rate, amortization_months, -loan_balance)

        # calculate balloon balance using closed-form formula
        if monthly_rate == 0:
            # calculate expected balloon if rate is equal to zero
            # implying monthly_principal = loan_balance / amortization_months
            # and expected_balloon = loan_balance - monthly_principal * amortizing_months_in_term 
            # which simplifies to formula below
            expected_balloon = loan_balance * (1 - amortizing_months_in_term/amortization_months)
        else:
            # calculate expected balloon if rate is nonzero
            # let loan_balance = L, monthly_rate = r, k = amortizing_months_in_term, and expected payment = P = closed-form annuity formula
            # then new_balance = old_balance*(1+r) - P
            # after k payments, balance = L(1+r)^k - P(1+r)^(k-1) - P(1+r)^(k-2) - ... - P(1+r) - P
            # simplifying, balance = L(1+r)^k - P[(1+r)^(k-1) + ... + (1+r) + 1]
            # note that the geometric series above equals: [(1+r)^(k-1) + ... + (1+r) + 1] = ((1+r)^k - 1)/r
            # so: balance = L(1+r)^k - P((1+r)^k - 1)/r
            # substituting P and simplifying gives the formula below
            expected_balloon = loan_balance * ((1 + monthly_rate)**amortization_months - (1 + monthly_rate)**amortizing_months_in_term) / ((1 + monthly_rate)**amortization_months - 1)

        # ensure all payments equal numpy-financial pmt function benchmark 
        if (mortgage_schedule.iloc[io_months:]["Payment"] - expected_payment).abs().gt(tolerance).any():
            findings.append("Amortizing payment does not match numpy-financial pmt function benchmark")
        # calculate balloon balance using numpy fv function as benchmark
        expected_balloon_np_benchmark = npf.fv(monthly_rate, amortizing_months_in_term, expected_payment, -loan_balance)
        if abs(mortgage_schedule.iloc[-1]["Balloon Payment"] - expected_balloon_np_benchmark) > tolerance:
            findings.append("Balloon payment does not match numpy-financial fv function benchmark")
    else:
        # no amortization leads to balloon payment = original loan_balance
        expected_balloon = loan_balance

    # ensure balloon payment from given mortgage schedule is equal to closed-form derivation of balloon payment
    if abs(mortgage_schedule.iloc[-1]["Balloon Payment"] - expected_balloon) > tolerance:
        findings.append("Balloon payment does not match closed-form remaining balance")

    # ensure the number of rows in the mortgage schedule is equal to the term_years in months
    if len(mortgage_schedule) != term_years*12:
        findings.append("The number of rows in the mortgage schedule does not equal the loan term in months")

    expected_months = pd.Series(range(1, len(mortgage_schedule) + 1), index=mortgage_schedule.index)
    is_wrong = mortgage_schedule["Month"] != expected_months

    # ensure the months within the mortgage schedule month column is 1,2,3,...,length of mortgage schedule 
    # and identifies incorrectly specified month
    if is_wrong.any():
        wrong_months = expected_months[is_wrong].tolist()
        wrong_months_str = ""
        for i in range(len(wrong_months)):
            if i != len(wrong_months) - 1:
                wrong_months_str += str(wrong_months[i]) + ", "
            else:
                wrong_months_str += str(wrong_months[i])
        findings.append("Month sequence is incorrect at month " + wrong_months_str)

    # ensure all balances are greater than or equal to zero
    if mortgage_schedule["Beginning Balance"].lt(0).any() or mortgage_schedule["Ending Balance"].lt(0).any() or mortgage_schedule["Principal"].lt(0).any():
        findings.append("At least one balance or principal payment is negative")

    # ensure the beginning balance in the first month is equal to the original loan_balance
    if abs(mortgage_schedule.iloc[0]["Beginning Balance"] - loan_balance) > tolerance:
        findings.append("Beginning balance at month 1 does not equal the original loan balance")

    # return T/F if validation pass/fail and a list containing all the invalid checks
    return len(findings) == 0, findings