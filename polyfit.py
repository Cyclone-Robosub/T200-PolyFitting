import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


class PolyFit:
    def __init__(self, file_path):
        self.file_path = file_path
        self.best_model, self.best_poly, self.best_degree, self.best_scaler = self.get_best_model()
        print(f"\nBest Model: {self.best_model}")
        print(f"Best Polynomial Features: {self.best_poly}")
        print(f"Best Degree: {self.best_degree}")
        print(f"Best Scaler: {self.best_scaler}")

    def get_best_model(self):

        xls = pd.ExcelFile(self.file_path)
        voltage_sheets = ['10 V', '12 V', '14 V', '16 V', '18 V', '20 V']
        data = []

        for sheet in voltage_sheets:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()
            V = float(sheet.split()[0])
            for _, row in df.iterrows():
                pwm = row['PWM (µs)']
                force = row['Force (Kg f)']
                data.append([V, pwm, force])

        data = np.array(data)
        V_vals = data[:, 0].reshape(-1, 1)
        PWM_vals = data[:, 1].reshape(-1, 1)
        Force_vals = data[:, 2]

        # Prepare feature matrix
        X = np.hstack((V_vals, PWM_vals))

        # Try different polynomial degrees
        best_mse = float("inf")
        best_model = None
        best_poly = None
        best_degree = None
        best_scaler = None

        for degree in range(2, 5):  # Trying degrees 2 to 4
            poly = PolynomialFeatures(degree=degree, interaction_only=False)
            X_poly = poly.fit_transform(X)

            # Normalize features
            scaler = StandardScaler()
            X_poly_scaled = scaler.fit_transform(X_poly)

            # Try different regression models
            models = {
                "Ridge Regression": Ridge(alpha=0.1),
                "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1)
            }

            for model_name, model in models.items():
                model.fit(X_poly_scaled, Force_vals)
                predicted_force = model.predict(X_poly_scaled)
                mse = mean_squared_error(Force_vals, predicted_force)

                if mse < best_mse:
                    best_mse = mse
                    best_model = model
                    best_poly = poly
                    best_degree = degree
                    best_scaler = scaler

        print(f"\nBest Model: {best_model}, Degree {best_degree}, MSE: {best_mse:.6f}")
        print(best_poly.get_feature_names_out(input_features=["V", "PWM"]))

        return best_model, best_poly, best_degree, best_scaler

    def estimate_thruster_force_optimized(self, V, PWM, model, poly, scaler):

        X = np.array([[V, PWM]])
        X_poly = poly.transform(X)
        X_poly_scaled = scaler.transform(X_poly)
        return model.predict(X_poly_scaled)[0]

    def estimate_thruster_force(self, V, PWM):
        return self.estimate_thruster_force_optimized(V, PWM, self.best_model, self.best_poly, self.best_scaler)

    def test_polynomial_fit(self):

        xls = pd.ExcelFile(self.file_path)
        voltage_sheets = ['10 V', '12 V', '14 V', '16 V', '18 V', '20 V']
        data = []

        for sheet in voltage_sheets:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()  # Clean column names
            V = float(sheet.split()[0])  # Extract voltage from sheet name
            for _, row in df.iterrows():
                pwm = row['PWM (µs)']
                force = row['Force (Kg f)']
                data.append([V, pwm, force])

        data = np.array(data)
        V_vals = data[:, 0]  # Voltage values
        PWM_vals = data[:, 1]  # PWM values
        Force_vals = data[:, 2]  # Actual Force values

        predicted_force = np.array([self.estimate_thruster_force(v, p) for v, p in zip(V_vals, PWM_vals)])
        mse = mean_squared_error(Force_vals, predicted_force)

        print(f"Mean Squared Error: {mse:.6f}")
        return mse

    def plot_absolute_error_by_pwm(self):

        xls = pd.ExcelFile(self.file_path)
        voltage_sheets = ['10 V', '12 V', '14 V', '16 V', '18 V', '20 V']

        plt.figure(figsize=(12, 8))
        for sheet in voltage_sheets:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()  # Clean column names
            V = float(sheet.split()[0])  # Extract voltage from sheet name
            pwm_values = df['PWM (µs)'].values
            force_actual = df['Force (Kg f)'].values
            force_predicted = np.array([self.estimate_thruster_force(V, pwm) for pwm in pwm_values])
            absolute_error = np.abs(force_actual - force_predicted)

            plt.plot(pwm_values, absolute_error, label=f"{V} V")

        plt.xlabel('PWM (µs)')
        plt.ylabel('Absolute Error kg(F)')
        plt.title('Absolute Error at Each Voltage Level')
        plt.legend()
        plt.grid()
        plt.show()

        print("Plots generated successfully.")

    def plot_relative_error_by_pwm(self):

        xls = pd.ExcelFile(self.file_path)
        voltage_sheets = ['10 V', '12 V', '14 V', '16 V', '18 V', '20 V']

        plt.figure(figsize=(12, 8))
        for sheet in voltage_sheets:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()  # Clean column names
            V = float(sheet.split()[0])  # Extract voltage from sheet name
            pwm_values = df['PWM (µs)'].values
            force_actual = df['Force (Kg f)'].values
            force_predicted = np.array([self.estimate_thruster_force(V, pwm) for pwm in pwm_values])
            relative_error = (force_actual - force_predicted) / force_actual

            plt.plot(pwm_values, relative_error, label=f"{V} V")

        plt.xlabel('PWM (µs)')
        plt.ylabel('Relative Error')
        plt.title('Relative Error at Each Voltage Level')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_predicted_vs_actual_force(self):

        xls = pd.ExcelFile(self.file_path)
        voltage_sheets = ['10 V', '12 V', '14 V', '16 V', '18 V', '20 V']

        for sheet in voltage_sheets:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()  # Clean column names
            V = float(sheet.split()[0])  # Extract voltage from sheet name
            pwm_values = df['PWM (µs)'].values
            force_actual = df['Force (Kg f)'].values
            force_predicted = np.array([self.estimate_thruster_force(V, pwm) for pwm in pwm_values])

            plt.figure(figsize=(8, 5))
            plt.plot(pwm_values, force_actual, label="Actual Force", linestyle='dashed')
            plt.plot(pwm_values, force_predicted, label="Predicted Force")
            plt.xlabel('PWM (µs)')
            plt.ylabel('Force (Kg f)')
            plt.title(f'Predicted vs Actual Force at {V} V')
            plt.legend()
            plt.grid()
            plt.show()

        print("Plots generated successfully.")


file_path = "T200-Public-Performance-Data-10-20V-September-2019.xlsx"
pf = PolyFit(file_path)
mse = pf.test_polynomial_fit()
pf.plot_absolute_error_by_pwm()
pf.plot_relative_error_by_pwm()
pf.plot_predicted_vs_actual_force()


