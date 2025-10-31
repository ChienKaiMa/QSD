# Grok 3 eliminates me.

import numpy as np


def inner_products_table(states):
    """
    Calculate inner products between quantum states with aligned table columns.

    Args:
        states: List of quantum states as numpy arrays (complex vectors)

    Returns:
        Matrix of inner products and prints a formatted table
    """
    n = len(states)
    # Initialize inner product matrix
    inner_prods = np.zeros((n, n), dtype=complex)

    # Calculate all inner products
    for i in range(n):
        for j in range(n):
            inner_prods[i][j] = np.vdot(states[i], states[j])

    # Fixed column width for numbers
    col_width = 18
    # Width for the first column (state labels)
    label_width = 10

    # Print formatted table
    print("\nInner Product Table:")
    # Calculate total width for separator line
    total_width = label_width + n * (col_width + 1) + 1
    print("-" * (total_width + 2))

    # Header row
    header = f"|{'States'.center(label_width)} |" + "".join(
        [f"{'State '+str(j)}".center(col_width) + "|" for j in range(n)]
    )
    print(header)
    print("-" * total_width)

    # Data rows
    for i in range(n):
        row = f"|  {'State '+str(i)}".center(label_width) + "  |"
        for j in range(n):
            val = inner_prods[i][j]
            # Format complex number nicely
            ## num_str = np.format_float_scientific(abs(val), precision=3)
            if abs(val.imag) < 1e-10:  # If imaginary part is negligible
                num_str = np.format_float_scientific(val.real, precision=3)
            else:
                num_str = f"{np.format_float_scientific(val.real, precision=3)}{'+' if val.imag >= 0 else ''}{np.format_float_scientific(val.imag, precision=3)}i"
            row += f"{num_str.center(col_width)}|"
        print(row)
    print("-" * (total_width + 2))

    return inner_prods


# Example usage with complex numbers
if __name__ == "__main__":
    # Define example quantum states with complex components
    state0 = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])  # (|0> + i|1>)/√2
    state1 = np.array([1, 0])  # |0>
    state2 = np.array([0, 1])  # |1>

    states = [state0, state1, state2]

    # Calculate and display inner products
    result = inner_products_table(states)
