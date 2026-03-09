import numpy as np

# ========== MEMBERSHIP FUNCTIONS ==========
def triangular_mf(x, a, b, c):
    if x <= a or x >= c: return 0
    elif a < x <= b: return (x - a) / (b - a)
    else: return (c - x) / (c - b)

def fuzzify_income(income):
    # Income in $1000s per year (realistic: 0-200k)
    params = {'Low': (0,0,50), 'Medium': (30,75,120), 'High': (80,150,200)}
    return {term: triangular_mf(income, a, b, c) for term, (a,b,c) in params.items()}

def fuzzify_credit(score):
    # Credit score (realistic: 300-850)
    params = {'Low': (300,300,580), 'Medium': (500,650,750), 'High': (680,780,850)}
    return {term: triangular_mf(score, a, b, c) for term, (a,b,c) in params.items()}

def fuzzify_stability(years):
    # Employment years (realistic: 0-40 years)
    params = {'Low': (0,0,2), 'Medium': (1,5,15), 'High': (8,20,40)}
    return {term: triangular_mf(years, a, b, c) for term, (a,b,c) in params.items()}

# ========== RULE EVALUATION ==========
def evaluate_rules(inc_f, cred_f, stab_f):
    rules = [
        ('High','High','High','High'), ('High','High','Medium','High'),
        ('High','Medium','High','High'), ('Medium','High','High','High'),
        ('Medium','Medium','Medium','Medium'), ('Medium','Low','Medium','Low'),
        ('Low','Medium','Medium','Low'), ('Low','Low','Low','Low'),
        ('Low','Medium','High','Medium'), ('Medium','High','Medium','Medium')
    ]
    
    strengths = []
    for inc, cred, stab, out in rules:
        strength = min(inc_f[inc], cred_f[cred], stab_f[stab])
        if strength > 0:
            strengths.append((out, strength))
    return strengths

# ========== MAMDANI ==========
def mamdani(strengths):
    if not strengths: return 0
    out_vals = {'Low':25, 'Medium':60, 'High':90}
    num = sum(s * out_vals[out] for out, s in strengths)
    den = sum(s for _, s in strengths)
    return num/den if den else 0

# ========== SUGENO ==========
def sugeno(strengths):
    if not strengths: return 0
    coeff = {'Low':30, 'Medium':65, 'High':95}
    num = sum(s * coeff[out] for out, s in strengths)
    den = sum(s for _, s in strengths)
    return num/den if den else 0

# ========== INTERPRET ==========
def interpret(val):
    if val >= 75: return "APPROVED"
    elif val >= 45: return "CONDITIONAL"
    else: return "REJECTED"

# ========== MAIN ==========
def evaluate_loan(income, credit, stability):
    inc_f = fuzzify_income(income)
    cred_f = fuzzify_credit(credit)
    stab_f = fuzzify_stability(stability)
    
    strengths = evaluate_rules(inc_f, cred_f, stab_f)
    m = mamdani(strengths)
    s = sugeno(strengths)
    
    return m, s, interpret((m+s)/2)

# ========== TEST WITH REALISTIC DATA ==========
def main():
    # Realistic loan applicants
    tests = [
        ["Software Engineer (Senior)", 145, 780, 8],
        ["Teacher (Entry)", 42, 650, 1.5],
        ["Restaurant Worker", 28, 580, 0.5],
        ["Doctor", 180, 820, 12],
        ["Recent Graduate", 55, 710, 0.2],
        ["Small Business Owner", 95, 690, 7],
        ["Construction Worker", 48, 600, 3],
        ["Retired", 60, 750, 0],  # 0 years stability counts as Low
        ["Freelancer (Irregular)", 65, 720, 2],
        ["Bank Manager", 120, 800, 15]
    ]
    
    print("\n" + "="*70)
    print("REALISTIC FUZZY LOAN APPROVAL SYSTEM")
    print("="*70)
    print(f"{'Applicant':<25} {'Income':<8} {'Credit':<8} {'Years':<6} {'Mamdani':<8} {'Sugeno':<8} {'Decision':<10}")
    print("-"*70)
    
    for name, inc, cred, stab in tests:
        m, s, dec = evaluate_loan(inc, cred, stab)
        print(f"{name[:24]:<25} ${inc:<7} {cred:<7} {stab:<6} {m:<8.1f} {s:<8.1f} {dec:<10}")

if __name__ == "__main__":
    main()