import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

class PolyFit:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.extract_force_pwm_voltage()

    def extract_force_pwm_voltage(self):
        xls = pd.ExcelFile(self.filename)
        sheet_names = xls.sheet_names

        data_list = []

        for sheet in sheet_names:
            if sheet != "READ ME FIRST":
                df = pd.read_excel(xls, sheet_name=sheet)
                df.columns = df.columns.str.strip()
                pwm = df["PWM (µs)"].values
                pwm = [value - 1500 for value in pwm]
                force = df["Force (Kg f)"].values
                voltage = np.full_like(pwm, float(sheet.split()[0]))
                data_list.extend(zip(force, pwm, voltage))

        dtype = [("force", "f4"), ("pwm", "i4"), ("voltage", "f4")]
        structured_array = np.array(data_list, dtype=dtype)
        return structured_array

    def polynomial_model(self, X,
                         a, b, c, d, e,
                         f, g, h, k, l,
                         m, n, o, p, q,
                         r, s, t, u, v,
                         w, x, y, z, aa,
                         ab, ac, ad, ae, af,
                         ag, ah, ak, al, am,
                         an, ao, ap, aq, ar):

        V, P = X
        coefs = [a, b, c, d, e,
                 f, g, h, k, l,
                 m, n, o, p, q,
                 r, s, t, u, v,
                 w, x, y, z, aa,
                 ab, ac, ad, ae, af,
                 ag, ah, ak, al, am,
                 an, ao, ap, aq, ar]
        num_coefs = len(coefs)
        n = 3
        terms = []
        for i in range(n + 1):
            for j in range(n + 1):
                terms.append(P**i * V**j)
        for i in range(num_coefs - len(terms)):
            terms.append(0)
        return sum([terms[i] * coefs[i] for i in range(num_coefs)])

    def evaluate_piecewise_force(self, V, P, coefficients):
        P = np.array(P)
        V = np.array(V)
        output = np.zeros_like(P, dtype=float)
        mask = P >= 28
        output[mask] = self.polynomial_model((V[mask], P[mask]), *coefficients)
        return output

    def evaluate_piecewise_force_lower(self, V, P, coefficients):
        P = np.array(P)
        V = np.array(V)
        output2 = np.zeros_like(P, dtype=float)
        mask = P <= -28
        output2[mask] = self.polynomial_model((V[mask], P[mask]), *coefficients)
        return output2

    def fit_polynomial_model_2(self):
        F = self.data["force"]
        V = self.data["voltage"]
        P = self.data["pwm"]

        mask = (P >= 28) & (P <= 400)
        F = F[mask]
        V = V[mask]
        P = P[mask]

        voltages = np.unique(V)
        anchor_P = np.full_like(voltages, 28)
        anchor_F = np.zeros_like(voltages)

        V = np.concatenate((V, voltages))
        P = np.concatenate((P, anchor_P))
        F = np.concatenate((F, anchor_F))

        popt, _ = curve_fit(self.polynomial_model, (V, P), F)
        return popt

    def fit_polynomial_model_lower(self):
        F = self.data["force"]
        V = self.data["voltage"]
        P = self.data["pwm"]

        mask = (P <= -28) & (P >= -400)
        F = F[mask]
        V = V[mask]
        P = P[mask]

        voltages = np.unique(V)
        anchor_P = np.full_like(voltages, -28)
        anchor_F = np.zeros_like(voltages)

        V = np.concatenate((V, voltages))
        P = np.concatenate((P, anchor_P))
        F = np.concatenate((F, anchor_F))

        popt, _ = curve_fit(self.polynomial_model, (V, P), F)
        return popt



    def evaluate_force_full_range(self, V, P, coeffs_main, coeffs_lower):
        P = np.array(P)
        V = np.array(V)
        output = np.zeros_like(P, dtype=float)

        mask_lower = P <= -28
        mask_main = P >= 28

        output[mask_lower] = self.polynomial_model((V[mask_lower], P[mask_lower]), *coeffs_lower)
        output[mask_main] = self.polynomial_model((V[mask_main], P[mask_main]), *coeffs_main)

        # Everything in between stays 0
        return output

    def plot_absolute_error(self, coefficients, coefficientsLower):
        data = self.data
        plt.figure(figsize=(12, 8))
        unique_voltages = np.unique(data["voltage"])
        for V in unique_voltages[:5]:
            subset = data[data["voltage"] == V]
            P = subset["pwm"]
            F_actual = subset["force"]
            F_predicted = self.evaluate_force_full_range(subset["voltage"], P, coefficients, coefficientsLower)
            abs_error = np.abs(F_actual - F_predicted)
            plt.plot(P, abs_error, label=f"{V} V")

        plt.xlabel('PWM - 1500 (µs)')
        plt.ylabel('Absolute Error (kg f)')
        plt.title('Absolute Error at Each Voltage Level')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_relative_error(self, coefficients, coefficientsLower):
        data = self.data
        plt.figure(figsize=(12,8))
        unique_voltages = np.unique(data["voltage"])
        for V in unique_voltages[:5]:
            subset = data[data["voltage"] == V]
            P = subset["pwm"]
            F_actual = subset["force"]
            F_predicted = self.evaluate_force_full_range(subset["voltage"], P, coefficients, coefficientsLower)
            rel_error = np.abs((F_actual - F_predicted) / F_actual)
            plt.plot(P, rel_error, label=f"{V} V")

        plt.xlabel('PWM - 1500 (µs)')
        plt.ylabel('Relative Error')
        plt.title('Relative Error at Each Voltage Level')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_actual_vs_modeled(self, coeffs_below=None, coeffs_above=None):
        data = self.data
        unique_voltages = np.unique(data["voltage"])
        for V in unique_voltages[:5]:
            subset = data[data["voltage"] == V]
            P = subset["pwm"]
            F_actual = subset["force"]
            V_vals = subset["voltage"]

            if coeffs_below is not None and coeffs_above is not None:
                F_predicted = self.evaluate_force_full_range(V_vals, P, coeffs_below, coeffs_above)
            else:
                F_predicted = self.evaluate_force_full_range(V_vals, P, coeffs_below, coeffs_above)

            plt.figure(figsize=(8, 5))
            plt.plot(P, F_actual, label="Actual Force", linestyle='dashed')
            plt.plot(P, F_predicted, label="Predicted Force")
            plt.xlabel('PWM - 1500 (µs)')
            plt.ylabel('Force (Kg f)')
            plt.title(f'Predicted vs Actual Force at {V} V')
            plt.legend()
            plt.grid()
            plt.show()

    def test_model_mse(self, coefficients):
        data = self.data
        F_actual = data["force"]
        V = data["voltage"]
        P = data["pwm"]
        F_predicted = self.evaluate_force_full_range(V, P, coefficients, coefficientsLower)
        print(coefficients)
        self.plot_absolute_error(coefficients, coefficientsLower)
        self.plot_relative_error(coefficients, coefficientsLower)
        self.plot_actual_vs_modeled(coefficients, coefficientsLower)
        return mean_squared_error(F_actual, F_predicted)

file_path = "T200-Public-Performance-Data-10-20V-September-2019.xlsx"

pf = PolyFit(file_path)
coefficients = pf.fit_polynomial_model_2()
coefficientsLower = pf.fit_polynomial_model_lower()
print(f"MSE: {pf.test_model_mse(coefficients)}\n")

terms = []
n = 3
for i in range(n + 1):
    for j in range(n + 1):
        terms.append(f"(P^{i})(V^{j})")

for i in range(len(terms)):
    print(f"{coefficients[i]} {terms[i]}")



terms = [f"(P^{i})(V^{j})" for i in range(4) for j in range(4)]

print("\nCoefficients for 16V zero band P >= 28 (Upper Region):")
for i, t in enumerate(terms):
    print(f"{coefficients[i]} {t}")

print("\nCoefficients for 16V zero band P <= -28 (Lower Region):")
for i, t in enumerate(terms):
    print(f"{coefficientsLower[i]} {t}")