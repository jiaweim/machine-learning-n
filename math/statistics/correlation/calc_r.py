from scipy.stats import pearsonr
import numpy as np

x = np.array([1.8, 1.3, 2.4, 1.5, 3.9, 2.1, 0.9, 1.4, 3, 4.6])  # GDP
y = np.array([604.4, 434.2, 544, 370.4, 742.3, 340.5, 232, 262.3, 441.9, 1157.7, ])  # CO2

r = pearsonr(x, y)
print(r)

x1 = np.array([1, 10, 5, 15, 3, 24, 30])  # Number of years out of school
y1 = np.array([12.5, 8.7, 14.6, 5.2, 9.9, 3.1, 2.7])  # Annual contribution
r1 = pearsonr(x1, y1)
print(r1)
