def format_complex_array(arr):
    # Format each complex number to 5 decimal places with explicit signs
    formatted = [f"{c.real:+.5f}{c.imag:+.5f}j" for c in arr]
    # Pad with empty string if array length is less than 4
    formatted += [""] * (4 - len(arr))
    # Join elements with spaces and enclose in brackets
    return f"[ {'  '.join(f'{s:>18}' for s in formatted)} ]"


def format_complex_array_color(arr):
    # ANSI color codes
    DIM = "\033[90m"  # Gray for zeros
    BOLD_GREEN = "\033[1;32m"  # Bold green for non-zeros
    RESET = "\033[0m"  # Reset formatting

    def format_number(x):
        # Format a single number (real or imaginary part)
        is_zero = abs(x) < 1e-10  # Handle floating-point precision
        color = DIM if is_zero else BOLD_GREEN
        return f"{color}{x:+.5f}{RESET}"

    # Format each complex number
    formatted = [
        f"{format_number(c.real)}{format_number(c.imag)}j" for c in arr
    ]
    # Pad with empty string if array length is less than 4
    formatted += [""] * (4 - len(arr))
    # Join elements with spaces and enclose in brackets
    return f"[ {'  '.join(f'{s:>18}' for s in formatted)} ]"


if __name__ == "__main__":
    print()
