from loan_model import create_loan_schedule
from validation import validate_loan_schedule

# creates a mortgage schedule with 
# original loan_balance = 10,000,000
# annual_rate = 6%
# term_years = 10 years
# amortization_years = 30 years (full amortization schedule period)
# io_months = 24 (interest-only periods)
df = create_loan_schedule(loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=30, io_months=24)

# Positive test: validation test of a well defined loan ---> should pass with (True, [])
assert validate_loan_schedule(mortgage_schedule=df, loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=30, io_months=24) == (True, [])

# Negative test: overwrite one month number with 99
# ---> should fail with "Month sequence is incorrect at month 6"
bad = df.copy()
bad.loc[5, "Month"] = 99
assert validate_loan_schedule(mortgage_schedule=bad, loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=30, io_months=24) == (False, ["Month sequence is incorrect at month 6"])

# Edge case: full-term interest-only loan (amortization_years = 0, io_months = full term).
# No principal is ever paid, so expect a 50,000 payment every month and a
# 10,000,000 balloon ---> should pass with (True, [])
full_io = create_loan_schedule(loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=0, io_months=120)
assert (full_io["Payment"] - 50_000).abs().max() < 0.01
assert abs(full_io.iloc[-1]["Balloon Payment"] - 10_000_000) < 0.01
assert validate_loan_schedule(mortgage_schedule=full_io, loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=0, io_months=120) == (True, [])

# Edge case: zero interest rate. Every payment is pure principal
# (10,000,000 / 360 = 27,777.78), so expect a 10,000,000 - 27,777.78 * 120 = 6,666,666.67 balloon
# ---> should pass with (True, [])
zero_rate = create_loan_schedule(loan_balance=10_000_000, annual_rate=0.0, term_years=10, amortization_years=30, io_months=0)
assert (zero_rate["Payment"] - 27_777.78).abs().max() < 0.01
assert abs(zero_rate.iloc[-1]["Balloon Payment"] - 6_666_666.67) < 0.01
assert validate_loan_schedule(mortgage_schedule=zero_rate, loan_balance=10_000_000, annual_rate=0.0, term_years=10, amortization_years=30, io_months=0) == (True, [])

# Negative test: raise month 2's beginning balance by 1,000 so it no longer
# equals month 1's ending balance ---> should fail with three findings:
# roll-forward, ending balance, and monthly interest
bad_roll = df.copy()
bad_roll.loc[1, "Beginning Balance"] = 10_001_000
assert validate_loan_schedule(mortgage_schedule=bad_roll, loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=30, io_months=24) == (False, [
    "Balance does not roll forward appropriately",
    "Ending balance does not equal beginning balance - principal",
    "Monthly interest does not equal beginning balance * monthly rate"
])

# Input validation test: 96 amortizing months but no amortization period is
# an impossible loan ---> create_loan_schedule should raise a ValueError
try:
    create_loan_schedule(loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=0, io_months=24)
    was_rejected = False
except ValueError:
    was_rejected = True
assert was_rejected

# Negative test: cut the full-term interest-only schedule down to 100 rows and move the
# balloon to the new last row ---> should fail with "The number of rows in the mortgage schedule does not equal the loan term in months"
short_io = full_io.iloc[:100].copy()
short_io.loc[short_io.index[-1], "Balloon Payment"] = short_io.iloc[-1]["Ending Balance"]
short_io["Final Ending Balance"] = short_io["Ending Balance"] - short_io["Balloon Payment"]
assert validate_loan_schedule(mortgage_schedule=short_io, loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=0, io_months=120) == (False, [
    "The number of rows in the mortgage schedule does not equal the loan term in months"
])

# Negative test: overstate the final balance and balloon by 5,000
# ---> should fail, and both independent balloon benchmarks should report it
bad_balloon = df.copy()
bad_balloon.loc[bad_balloon.index[-1], ["Ending Balance", "Balloon Payment"]] += 5000
is_valid, findings = validate_loan_schedule(mortgage_schedule=bad_balloon, loan_balance=10_000_000, annual_rate=0.06, term_years=10, amortization_years=30, io_months=24)
assert is_valid == False
assert "Balloon payment does not match closed-form remaining balance" in findings
assert "Balloon payment does not match numpy-financial fv function benchmark" in findings

print("All 8 tests passed")
