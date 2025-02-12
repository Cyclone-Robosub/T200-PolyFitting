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

        # Initialize an empty list to store data
        data_list = []

        # Process each voltage sheet (excluding the "READ ME FIRST" sheet)
        for sheet in sheet_names:
            if sheet != "READ ME FIRST":
                df = pd.read_excel(xls, sheet_name=sheet)

                # Strip spaces from column names
                df.columns = df.columns.str.strip()

                # Extract relevant columns
                pwm = df["PWM (µs)"].values
                pwm = [value - 1500 for value in pwm]
                force = df["Force (Kg f)"].values
                voltage = np.full_like(pwm, float(sheet.split()[0]))  # Extract voltage from sheet name

                # Append data as tuples
                data_list.extend(zip(force, pwm, voltage))

        # Convert to structured NumPy array
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
                         an, ao, ap, aq, ar
                         ):

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

    def fit_polynomial_model_2(self):

        # Extract variables
        F = self.data["force"]
        V = self.data["voltage"]
        P = self.data["pwm"]

        # Fit model using non-linear least squares
        popt, _ = curve_fit(self.polynomial_model, (V, P), F)
        return popt


    def plot_absolute_error(self, coefficients):

        data = self.data
        plt.figure(figsize=(12, 8))
        unique_voltages = np.unique(data["voltage"])
        for V in unique_voltages[:5]:
            subset = data[data["voltage"] == V]
            P = subset["pwm"]
            F_actual = subset["force"]
            F_predicted = self.polynomial_model((V, P), *coefficients)
            abs_error = np.abs(F_actual - F_predicted)
            plt.plot(P, abs_error, label=f"{V} V")

        plt.xlabel('PWM - 1500 (µs)')
        plt.ylabel('Absolute Error (kg f)')
        plt.title('Absolute Error at Each Voltage Level')
        plt.legend()
        plt.grid()
        plt.show()


    def plot_relative_error(self, coefficients):
        """
        Plots the relative error at each voltage.
        """
        data = self.data
        plt.figure(figsize=(12,8))
        unique_voltages = np.unique(data["voltage"])
        for V in unique_voltages[:5]:
            subset = data[data["voltage"] == V]
            P = subset["pwm"]
            F_actual = subset["force"]
            F_predicted = self.polynomial_model((V, P), *coefficients)
            rel_error = np.abs((F_actual - F_predicted) / F_actual)
            plt.plot(P, rel_error, label=f"{V} V")

        plt.xlabel('PWM - 1500 (µs)')
        plt.ylabel('Relative Error')
        plt.title('Relative Error at Each Voltage Level')
        plt.legend()
        plt.grid()
        plt.show()





    def plot_actual_vs_modeled(self, coefficients):

        data = self.data
        unique_voltages = np.unique(data["voltage"])
        for V in unique_voltages[:5]:
            subset = data[data["voltage"] == V]
            P = subset["pwm"]
            F_actual = subset["force"]
            F_predicted = self.polynomial_model((subset["voltage"], P), *coefficients)


            plt.figure(figsize=(8, 5))
            plt.plot(P, F_actual, label="Actual Force", linestyle='dashed')
            plt.plot(P, F_predicted, label="Predicted Force")
            plt.xlabel('PWM - 1500 (µs)')
            plt.ylabel('Force (Kg f)')
            plt.title(f'Predicted vs Actual Force at {V} V')
            plt.legend()
            plt.grid()
            plt.show()

            # plt.figure()
            # plt.plot(P, F_actual, "bo", label="Actual Force")
            # plt.plot(P, F_predicted, "r-", label="Modeled Force")
            # plt.xlabel("PWM (µs)")
            # plt.ylabel("Force (Kg f)")
            # plt.title(f"Actual vs Modeled Force at {V}V")
            # plt.legend()
            # plt.show()


    def test_model_mse(self, coefficients):
        # Extract variables
        data = self.data
        F_actual = data["force"]
        V = data["voltage"]
        P = data["pwm"]

        # Predict force using the model
        F_predicted = self.polynomial_model((V, P), *coefficients)
        print(coefficients)

        self.plot_absolute_error(coefficients)
        self.plot_relative_error(coefficients)
        self.plot_actual_vs_modeled(coefficients)
        # Compute and return MSE
        return mean_squared_error(F_actual, F_predicted)

file_path = "T200-Public-Performance-Data-10-20V-September-2019.xlsx"

pf = PolyFit(file_path)
coefficients = pf.fit_polynomial_model_2()
print(f"MSE: {pf.test_model_mse(coefficients)}\n")

terms = []
n = 3
for i in range(n + 1):
    for j in range(n + 1):
        terms.append(f"(P^{i})(V^{j})")

for i in range(len(terms)):
    print(f"{coefficients[i]} {terms[i]}")


