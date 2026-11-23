
import numpy as np
import matplotlib.pyplot as plt

years = np.array([
    1971, 1972, 1974, 1978, 1982, 1985, 1989, 1993,
    1995, 1997, 1999, 2000, 2001, 2002, 2003, 2006,
    2008, 2011, 2012, 2013, 2015, 2017, 2019, 2021
], dtype=float)

transistors = np.array([
    2300, 3500, 6000, 29000, 134000, 275000, 1200000,
    3100000, 5500000, 7500000, 9500000, 42000000,
    25000000, 55000000, 220000000, 291000000,
    731000000, 1160000000, 1400000000, 1400000000,
    1900000000, 1750000000, 3900000000, 9800000000
], dtype=float)

t = years - years[0]
log_n = np.log(transistors)

linear_coeff = np.polyfit(t, log_n, 1)
quadratic_coeff = np.polyfit(t, log_n, 2)

k, log_a = linear_coeff
q, b, log_a2 = quadratic_coeff

a = np.exp(log_a)
a2 = np.exp(log_a2)

def exponential_model(x):
    return a * np.exp(k * x)

def exponential_rate(x):
    return k * exponential_model(x)

def exponential_acceleration(x):
    return k**2 * exponential_model(x)

def variable_rate_model(x):
    return a2 * np.exp(q * x**2 + b * x)

def variable_rate_derivative(x):
    return (2 * q * x + b) * variable_rate_model(x)

def variable_rate_acceleration(x):
    growth_rate = 2 * q * x + b
    return (2 * q + growth_rate**2) * variable_rate_model(x)

def r_squared(actual, predicted):
    return 1 - np.sum((actual - predicted)**2) / np.sum(
        (actual - np.mean(actual))**2
    )

historical_rate = np.gradient(transistors, years)
historical_acceleration = np.gradient(historical_rate, years)

predicted_exponential = exponential_model(t)
predicted_variable = variable_rate_model(t)

exponential_r2 = r_squared(transistors, predicted_exponential)
variable_r2 = r_squared(transistors, predicted_variable)

linear_log_r2 = r_squared(
    log_n, np.polyval(linear_coeff, t)
)

quadratic_log_r2 = r_squared(
    log_n, np.polyval(quadratic_coeff, t)
)

x = np.linspace(0, t[-1], 600)
future_x = np.linspace(0, 80, 600)

fig, axes = plt.subplots(3, 2, figsize=(16, 15))

axes[0, 0].scatter(years, transistors, label="Historical data")
axes[0, 0].plot(years[0] + x, exponential_model(x), label="Exponential model")
axes[0, 0].plot(years[0] + x, variable_rate_model(x), label="Variable-rate model")
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("Layer 0: Computing Capacity")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Transistor Count")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].scatter(years, historical_rate, label="Historical first derivative")
axes[0, 1].plot(years[0] + x, exponential_rate(x), label="Exponential derivative")
axes[0, 1].plot(years[0] + x, variable_rate_derivative(x), label="Variable-rate derivative")
axes[0, 1].set_yscale("symlog")
axes[0, 1].set_title("Layer 1: Rate of Change")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Transistors per Year")
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].scatter(years, historical_acceleration, label="Historical second derivative")
axes[1, 0].plot(years[0] + x, exponential_acceleration(x), label="Exponential model")
axes[1, 0].plot(years[0] + x, variable_rate_acceleration(x), label="Variable-rate model")
axes[1, 0].set_yscale("symlog")
axes[1, 0].set_title("Layer 2: Acceleration of Change")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Transistors per Year Squared")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].scatter(t, log_n, label="Historical logarithmic data")
axes[1, 1].plot(x, np.polyval(linear_coeff, x), label="Linear logarithmic model")
axes[1, 1].plot(x, np.polyval(quadratic_coeff, x), label="Quadratic logarithmic model")
axes[1, 1].set_title("Logarithmic Transformation")
axes[1, 1].set_xlabel("Years Since 1971")
axes[1, 1].set_ylabel("Natural Log of Transistor Count")
axes[1, 1].legend()
axes[1, 1].grid(True)

axes[2, 0].plot(x, np.full_like(x, k), label="Constant relative growth rate")
axes[2, 0].plot(x, 2 * q * x + b, label="Variable relative growth rate")
axes[2, 0].set_title("Layer 3: Evolution of the Growth Rate")
axes[2, 0].set_xlabel("Years Since 1971")
axes[2, 0].set_ylabel("Relative Growth Rate per Year")
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(future_x, variable_rate_model(future_x), label="Variable-rate model")
axes[2, 1].plot(future_x, exponential_model(future_x), label="Constant-rate exponential model")
axes[2, 1].axvline(t[-1], linestyle="--", label="End of historical data")
axes[2, 1].set_yscale("log")
axes[2, 1].set_title("Long-Term Model Comparison")
axes[2, 1].set_xlabel("Years Since 1971")
axes[2, 1].set_ylabel("Modeled Transistor Count")
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

print("\nMULTILAYER COMPUTING EVOLUTION ANALYSIS")
print("=" * 55)

print(f"Historical period: {int(years[0])}-{int(years[-1])}")
print(f"Initial transistor count: {transistors[0]:,.0f}")
print(f"Final transistor count: {transistors[-1]:,.0f}")

print("\nFIRST-ORDER EXPONENTIAL MODEL")
print(f"N(t) = {a:.4e} * exp({k:.6f}t)")
print(f"Relative growth rate: {k * 100:.3f}% per year")
print(f"Doubling time: {np.log(2) / k:.2f} years")

print("\nVARIABLE-RATE MODEL")
print(f"N(t) = {a2:.4e} * exp({b:.6f}t + {q:.8f}t^2)")
print(f"Initial relative growth rate: {b * 100:.3f}% per year")
print(f"Change in relative growth rate: {2 * q:.8f} per year squared")
print(f"Relative growth rate in final year: {(b + 2 * q * t[-1]) * 100:.3f}% per year")

print("\nMODEL COMPARISON")
print(f"Exponential model R-squared: {exponential_r2:.4f}")
print(f"Variable-rate model R-squared: {variable_r2:.4f}")
print(f"Linear logarithmic model R-squared: {linear_log_r2:.4f}")
print(f"Quadratic logarithmic model R-squared: {quadratic_log_r2:.4f}")

print("\nMATHEMATICAL LAYERS")
print("Layer 0: N(t)       = Computing capacity")
print("Layer 1: N'(t)      = Rate of change")
print("Layer 2: N''(t)     = Acceleration of change")
print("Layer 3: N'''(t)    = Change in acceleration")

print("\nINTERPRETATION")
if variable_r2 > exponential_r2:
    print("The variable-rate model fits the historical data better.")
else:
    print("The exponential model fits the historical data better.")

if q > 0:
    print("The fitted relative growth rate is increasing.")
elif q < 0:
    print("The fitted relative growth rate is decreasing.")
else:
    print("The fitted relative growth rate is constant.")

print("\nA higher R-squared indicates a better fit, not proof of causation.")
print("Transistor count is a proxy, not a direct measurement of computing power.")
print("The models are mathematical approximations, not future predictions.")
