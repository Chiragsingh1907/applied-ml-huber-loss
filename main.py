import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
import os

# Create a folder for the figures if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Generate synthetic data (A simple linear relationship)
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y_clean = 2 * X.ravel() + 1 + np.random.normal(0, 1, 20)

# Create variations of the data for the frames
data_stages = [
    (y_clean, "Frame 1: Clean Data (No Outliers)"),
    (np.copy(y_clean), "Frame 2: Single Minor Outlier"),
    (np.copy(y_clean), "Frame 3: Moderate Outliers Added"),
    (np.copy(y_clean), "Frame 4: Extreme Outliers Added")
]

# Add progressively worse outliers
data_stages[1][0][18] += 15 
data_stages[2][0][18] += 20
data_stages[2][0][2] -= 15
data_stages[3][0][18] += 40
data_stages[3][0][2] -= 30
data_stages[3][0][10] += 35

# Generate the plots
for i, (y, title) in enumerate(data_stages):
    plt.figure(figsize=(8, 6))
    
    # Fit Standard MSE (Linear Regression)
    mse_model = LinearRegression()
    mse_model.fit(X, y)
    
    # Fit Huber Regressor
    huber_model = HuberRegressor(epsilon=1.35)
    huber_model.fit(X, y)
    
    # Plotting
    plt.scatter(X, y, color='black', label='Data Points', s=50, zorder=3)
    plt.plot(X, mse_model.predict(X), color='red', linestyle='--', linewidth=2, label='MSE Line (Skewed)')
    plt.plot(X, huber_model.predict(X), color='blue', linewidth=2, label='Huber Line (Robust)')
    
    plt.title(title, fontsize=14, fontweight='bold', color='#1F3864')
    plt.xlabel('Input Feature (X)')
    plt.ylabel('Target Value (y)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the frame
    plt.savefig(f'figures/frame{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Success! Upload the 4 images in the 'figures' folder to Overleaf.")