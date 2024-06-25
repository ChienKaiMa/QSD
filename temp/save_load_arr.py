# Example code
import numpy as np

if __name__ == "__main__":
    num_points = 40
    x_axis = np.array(range(1, num_points)) * (1 / num_points)
    sqrt_arr = [np.sqrt(i) for i in x_axis]

    fname = "sqrt.csv"
    np.savetxt(
        fname, sqrt_arr, fmt="%.16f", delimiter=","
    )
    test_arr = np.loadtxt(fname=fname)
    print(test_arr)
    quit()
