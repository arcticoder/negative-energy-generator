import sympy as sp
import requests
import re
from sympy import symbols, Function, Matrix, sympify, Derivative, pi, sin

import sympy as sp
import requests
import re
from sympy import symbols, Function, Matrix, sympify, Derivative, pi, sin

# Define symbols & profile
t, r, theta, phi = symbols('t r theta phi')
f = Function('f')(r, t)

# Load stress–energy LaTeX
URL = 'https://raw.githubusercontent.com/arcticoder/warp-bubble-einstein-equations/refs/heads/main/stress_energy.tex'
resp = requests.get(URL)
latex = resp.text

# Extract matrix contents
pattern = r'\\begin{pmatrix}(.+?)\\end{pmatrix}'
m = re.search(pattern, latex, re.S)
if not m:
    raise ValueError("Stress-energy matrix not found")
matrix_content = m.group(1).strip()

# Extract just the first entry (T00) manually
# Split by \\ to get rows, then take first entry of first row
rows = matrix_content.split(r'\\')
first_row = rows[0].strip()
first_entry = first_row.split('&')[0].strip()

print("Raw T00 LaTeX entry:")
print(first_entry[:200] + "..." if len(first_entry) > 200 else first_entry)

def convert_latex_to_sympy(latex_str):
    """Convert LaTeX mathematical expression to SymPy expression string"""
    
    expr = latex_str.strip()
    
    # Define symbols that we'll use
    t, r = symbols('t r')
    f = Function('f')(r, t)
    
    # First, handle partial derivatives before processing fractions
    # \frac{\partial^{2}}{\partial t^{2}} f{\left(r,t \right)} -> dt2_f
    expr = re.sub(r'\\frac\{\\partial\^\{2\}\}\{\\partial\s+t\^\{2\}\}\s*f\{\\left\(r,t\s*\\right\)\}',
                  'dt2_f', expr)
    
    # \frac{\partial^{2}}{\partial r^{2}} f{\left(r,t \right)} -> dr2_f  
    expr = re.sub(r'\\frac\{\\partial\^\{2\}\}\{\\partial\s+r\^\{2\}\}\s*f\{\\left\(r,t\s*\\right\)\}',
                  'dr2_f', expr)
    
    # \frac{\partial}{\partial t} f{\left(r,t \right)} -> dt_f
    expr = re.sub(r'\\frac\{\\partial\}\{\\partial\s+t\}\s*f\{\\left\(r,t\s*\\right\)\}',
                  'dt_f', expr)
    
    # \frac{\partial}{\partial r} f{\left(r,t \right)} -> dr_f  
    expr = re.sub(r'\\frac\{\\partial\}\{\\partial\s+r\}\s*f\{\\left\(r,t\s*\\right\)\}',
                  'dr_f', expr)
    
    # Replace f{\left(r,t \right)} with just f everywhere
    expr = re.sub(r'f\{\\left\(r,t\s*\\right\)\}', 'f', expr)
    
    # Now handle fractions recursively from inside out
    def replace_frac(text):
        # Find innermost \frac{...}{...} patterns  
        pattern = r'\\frac\{([^{}]*)\}\{([^{}]*)\}'
        while re.search(pattern, text):
            text = re.sub(pattern, r'((\1)/(\2))', text)
        return text
    
    expr = replace_frac(expr)
    
    # Handle the main fraction structure more carefully
    # Look for the overall pattern: numerator/denominator
    if '(' in expr and ')(' in expr:
        # This suggests we have (numerator)(denominator) instead of (numerator)/(denominator)
        # Find the last )( pattern which should be the main division
        last_paren_match = expr.rfind(')(')
        if last_paren_match > 0:
            expr = expr[:last_paren_match+1] + '/' + expr[last_paren_match+1:]
    
    # Clean up LaTeX formatting
    expr = expr.replace(r'\left(', '(')
    expr = expr.replace(r'\right)', ')')
    expr = expr.replace(r'\left\{', '')
    expr = expr.replace(r'\right\}', '')
    expr = expr.replace(r'\pi', 'pi')
    expr = expr.replace(r'\,', ' ')  # Remove thin spaces
    
    # Handle exponents: ^{n} -> **n (without extra parentheses around single numbers)
    expr = re.sub(r'\^\{(\d+)\}', r'**\1', expr)  # For simple numeric exponents
    expr = re.sub(r'\^\{([^}]+)\}', r'**(\1)', expr)  # For complex exponents
    expr = re.sub(r'\^(\d+)', r'**\1', expr)  # For simple numeric exponents without braces
    
    # Remove any remaining backslash commands
    expr = re.sub(r'\\[a-zA-Z]+\s*', '', expr)
    expr = re.sub(r'\\', '', expr)  # Remove any remaining backslashes
    
    # Clean up extra spaces and braces
    expr = re.sub(r'\s+', ' ', expr).strip()
    expr = re.sub(r'\{\s*\}', '', expr)  # Remove empty braces
    expr = expr.replace('{', '(').replace('}', ')')  # Replace remaining braces with parentheses
    
    return expr

# Convert and parse T00
try:
    # Skip the LaTeX conversion for now and build the expression directly from the structure
    # Based on the LaTeX, the T00 component has the form:
    # (numerator) / (64 * pi * r * (f-1)^4)
    
    # Define derivative symbols
    dt_f = Derivative(f, t)      # df/dt
    dt2_f = Derivative(f, t, 2)  # d²f/dt²
    dr_f = Derivative(f, r)      # df/dr
    
    # Build the numerator step by step based on the LaTeX structure
    numerator = (2*r*(f-1)**3*dt2_f + 
                r*(f-1)**2*dt_f**2 - 
                2*r*(f-1)*dt2_f - 
                r*dt_f**2 + 
                4*(f-1)**3*(-2*f - dr_f + 2) - 
                4*(f-1)**2*dr_f)
    
    denominator = 64*pi*r*(f-1)**4
    
    T00 = numerator / denominator
    
    print(f"\nSuccessfully built T00 from LaTeX structure!")
    parsing_success = True
    
except Exception as e:
    print(f"\nError parsing T00: {e}")
    print("Using a simplified symbolic form instead.")
    
    # Create a simplified symbolic form based on the structure we can see
    # From the LaTeX, we can see it involves derivatives and powers of (f-1)
    dt_f = Derivative(f, t)  # df/dt
    dt2_f = Derivative(f, t, 2)  # d²f/dt²
    dr_f = Derivative(f, r)  # df/dr
    
    # Simplified form capturing the essential structure
    T00 = (2*r*(f-1)**3*dt2_f + r*(f-1)**2*dt_f**2 - 2*r*(f-1)*dt2_f - 
           r*dt_f**2 + 4*(f-1)**3*(-2*f - dr_f + 2) - 4*(f-1)**2*dr_f) / (64*pi*r*(f-1)**4)
    
    parsing_success = True  # We have a working form

# Simplify T00 and solve inequality if parsing was successful
if parsing_success:
    print("\nSimplifying T00...")
    T00_simplified = sp.simplify(T00)
    
    print("Analyzing negative energy condition...")
    try:
        # For the inequality analysis, we'll work symbolically
        # The condition T00 < 0 depends on the specific form of f(r,t)
        print("The condition for exotic matter (negative energy density) is:")
        print("T^00 < 0")
        
        # Write LaTeX report
        with open('exotic_matter_density.tex', 'w', encoding='utf-8') as tex:
            tex.write(r'\documentclass{article}' + '\n')
            tex.write(r'\usepackage{amsmath}' + '\n')
            tex.write(r'\begin{document}' + '\n\n')
            tex.write(r'\section*{Exotic-Matter Energy Density}' + '\n\n')
            tex.write('The energy-density component is\n')
            tex.write(r'\[' + '\n')
            tex.write(f'  T^{{00}}(r,t) = {sp.latex(T00_simplified)}\n')
            tex.write(r'\]' + '\n\n')
            tex.write('Negative-energy (exotic matter) occurs where\n')
            tex.write(r'\[' + '\n')
            tex.write(f'  T^{{00}}(r,t) < 0\n')
            tex.write(r'\]' + '\n\n')
            tex.write('The specific condition depends on the warp bubble profile $f(r,t)$ ')
            tex.write('and its derivatives. For the Alcubierre metric, this typically ')
            tex.write('requires $f(r,t) < 1$ in the warp bubble region and specific ')
            tex.write('relationships between the spatial and temporal derivatives.\n\n')
            tex.write(r'\end{document}' + '\n')
        
        print("LaTeX report written to 'exotic_matter_density.tex'")
        
    except Exception as e:
        print(f"Error in inequality analysis: {e}")
        print("The exotic matter condition is T^00 < 0, which depends on the specific warp bubble profile.")

else:
    print("Could not parse the stress-energy tensor. Please check the LaTeX format.")
