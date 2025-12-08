import numpy as np
from simulation.test_precision import run_test
from plotting.plot_precision import plot_precision

def main():
    # 試す dt 値
    dt_list = np.array([1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])

    err_euler = []
    err_rk4 = []

    for dt in dt_list:
        print(f"Testing dt = {dt}")

        e_err = run_test(dt, method="euler")
        r_err = run_test(dt, method="rk4")

        err_euler.append(e_err)
        err_rk4.append(r_err)

    err_euler = np.array(err_euler)
    err_rk4  = np.array(err_rk4)

    plot_precision(dt_list, err_euler, err_rk4)

if __name__ == "__main__":
    main()
