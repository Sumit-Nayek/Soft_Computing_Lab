import numpy as np
import matplotlib.pyplot as plt

class SimpleLoanFIS:
    def __init__(self):
        # Define membership function parameters
        # All ranges: 0-100 (scaled)
        self.income_params = {'Low': (0, 0, 40), 'Medium': (30, 50, 70), 'High': (60, 100, 100)}
        self.credit_params = {'Low': (0, 0, 40), 'Medium': (30, 50, 70), 'High': (60, 100, 100)}
        self.stability_params = {'Low': (0, 0, 40), 'Medium': (30, 50, 70), 'High': (60, 100, 100)}
        
        # Rules: (Income, Credit, Stability) -> Approval
        self.rules = [
            ('High', 'High', 'High', 'High'),
            ('High', 'High', 'Medium', 'High'),
            ('High', 'Medium', 'High', 'High'),
            ('Medium', 'High', 'High', 'High'),
            ('Medium', 'Medium', 'Medium', 'Medium'),
            ('Medium', 'Low', 'Medium', 'Low'),
            ('Low', 'Medium', 'Medium', 'Low'),
            ('Low', 'Low', 'Low', 'Low'),
            ('Low', 'Medium', 'High', 'Medium'),
            ('Medium', 'High', 'Medium', 'Medium')
        ]
        
        # Sugeno coefficients for each output class
        self.sugeno_coeff = {'Low': 20, 'Medium': 50, 'High': 80}

    def triangular_mf(self, x, a, b, c):
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        return 0

    def fuzzify(self, value, params):
        """Convert crisp value to fuzzy membership degrees"""
        degrees = {}
        for term, (a, b, c) in params.items():
            degrees[term] = self.triangular_mf(value, a, b, c)
        return degrees

    def evaluate_rules(self, income_fuzzy, credit_fuzzy, stability_fuzzy):
        """Evaluate all rules using min operator for AND"""
        rule_strengths = []
        
        for inc, cred, stab, output in self.rules:
            # Strength = min of antecedents
            strength = min(income_fuzzy[inc], credit_fuzzy[cred], stability_fuzzy[stab])
            if strength > 0:
                rule_strengths.append((output, strength))
        
        return rule_strengths

    # ========== MAMDANI METHOD ==========
    def mamdani_aggregate(self, rule_strengths):
        """Aggregate rules using max operator"""
        # Simplified: just return weighted average of output strengths
        output_sum = 0
        strength_sum = 0
        
        for output, strength in rule_strengths:
            if output == 'Low':
                output_sum += strength * 25
            elif output == 'Medium':
                output_sum += strength * 50
            else:  # High
                output_sum += strength * 75
            strength_sum += strength
        
        if strength_sum == 0:
            return 0
        
        # Defuzzify using centroid approximation
        return output_sum / strength_sum

    # ========== SUGENO METHOD ==========
    def sugeno_compute(self, rule_strengths):
        """Sugeno weighted average"""
        numerator = 0
        denominator = 0
        
        for output, strength in rule_strengths:
            numerator += strength * self.sugeno_coeff[output]
            denominator += strength
        
        if denominator == 0:
            return 0
        
        return numerator / denominator

    def evaluate_loan(self, income, credit, stability, method='both'):
        """Main evaluation function"""
        # Scale inputs to 0-100 if needed
        income = min(income, 100)
        credit = min(credit, 100)
        stability = min(stability, 100)
        
        # Fuzzification
        inc_fuzzy = self.fuzzify(income, self.income_params)
        cred_fuzzy = self.fuzzify(credit, self.credit_params)
        stab_fuzzy = self.fuzzify(stability, self.stability_params)
        
        # Rule evaluation
        rule_strengths = self.evaluate_rules(inc_fuzzy, cred_fuzzy, stab_fuzzy)
        
        results = {}
        if method in ['mamdani', 'both']:
            results['mamdani'] = self.mamdani_aggregate(rule_strengths)
        if method in ['sugeno', 'both']:
            results['sugeno'] = self.sugeno_compute(rule_strengths)
        
        return results

    def interpret_result(self, value):
        """Convert numerical result to linguistic interpretation"""
        if value < 30:
            return "LOW (Reject)"
        elif value < 60:
            return "MEDIUM (Consider with conditions)"
        else:
            return "HIGH (Approve)"

# ========== DEMONSTRATION ==========
def demonstrate():
    fis = SimpleLoanFIS()
    
    # Test cases
    test_cases = [
        {"name": "Excellent Applicant", "income": 90, "credit": 85, "stability": 95},
        {"name": "Average Applicant", "income": 55, "credit": 60, "stability": 50},
        {"name": "Poor Applicant", "income": 30, "credit": 25, "stability": 20},
        {"name": "Mixed Applicant", "income": 80, "credit": 40, "stability": 70},
    ]
    
    print("=" * 60)
    print("FUZZY LOAN APPROVAL SYSTEM")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  Income: {case['income']}, Credit: {case['credit']}, Stability: {case['stability']}")
        
        results = fis.evaluate_loan(case['income'], case['credit'], case['stability'])
        
        if 'mamdani' in results:
            m_val = results['mamdani']
            print(f"  Mamdani: {m_val:.1f} - {fis.interpret_result(m_val)}")
        
        if 'sugeno' in results:
            s_val = results['sugeno']
            print(f"  Sugeno:  {s_val:.1f} - {fis.interpret_result(s_val)}")
        
        if 'mamdani' in results and 'sugeno' in results:
            diff = abs(results['mamdani'] - results['sugeno'])
            print(f"  Difference: {diff:.1f} points")

    # Show membership functions
    print("\n" + "=" * 60)
    print("MEMBERSHIP FUNCTIONS VISUALIZATION")
    print("=" * 60)
    
    x = np.linspace(0, 100, 200)
    plt.figure(figsize=(10, 6))
    
    # Plot membership functions for one variable (same for all)
    for term, (a, b, c) in fis.income_params.items():
        y = [fis.triangular_mf(xi, a, b, c) for xi in x]
        plt.plot(x, y, label=f'{term} Income')
    
    plt.xlabel('Input Value')
    plt.ylabel('Membership Degree')
    plt.title('Fuzzy Membership Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    demonstrate()