import pandas as pd

df = pd.read_csv(r"C:\Users\ASUS\Documents\Housing.csv")  # or whatever your file name is
# Binary columns: Yes/No → 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# For 'furnishingstatus' → One-Hot Encoding
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

from sklearn.model_selection import train_test_split

# Features and Target
X_simple= df[['area']]  # simple regression — only one feature
y = df['price']
# Multiple regression (all features)
X_multi = df.drop('price', axis=1)

# Splits
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

# Simple Linear Regression
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# Multiple Linear Regression
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Simple Predictions
y_pred_s = simple_model.predict(X_test_s)

print("----- Simple Linear Regression -----")
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R² Score:", r2_score(y_test_s, y_pred_s))

# Multiple Predictions
y_pred_m = multi_model.predict(X_test_m)

print("\n----- Multiple Linear Regression -----")
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R² Score:", r2_score(y_test_m, y_pred_m))
import matplotlib.pyplot as plt

# Plot for Simple Linear Regression
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual Price')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Price vs Area")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted Prices
plt.scatter(y_test_m, y_pred_m, color='green')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted Prices")
plt.grid(True)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # identity line
plt.show()

# Residuals
residuals = y_test_m - y_pred_m

plt.scatter(y_pred_m, residuals, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Multiple Regression)")
plt.grid(True)
plt.show()
# Simple model coefficients
print("Simple Regression Coefficient (Slope):", simple_model.coef_[0])
print("Simple Regression Intercept:", simple_model.intercept_)

# Multiple model coefficients
coeff_df = pd.DataFrame(multi_model.coef_, X_multi.columns, columns=['Coefficient'])
print("\nMultiple Regression Coefficients:\n", coeff_df)
