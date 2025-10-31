# test_contour
# Written by Grok

import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Tkagg")

# Create data for the contour plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create the contour plot
plt.figure(figsize=(10, 8))

# contour or contourf
contour = plt.contour(X, Y, Z, levels=15, cmap='viridis')

# Add labels and title
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('2D Contour Plot of sin(sqrt(x^2 + y^2))')

# Add a color bar to show the scale of Z
plt.colorbar(contour)

# Show the plot
plt.savefig("contour_plot.png")