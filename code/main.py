import numpy as np
import matplotlib.pyplot as plt

P = np.array([[2.0, 0.0, 1.0],
             [1.08, 1.68, 2.38],
             [-0.83, 1.82, 2.49],
             [-1.97, 0.28, 2.15],
              [-1.31, -1.51, 2.59],
              [0.57, -1.91, 4.32]])

T = np.array([1, 2, 3, 4, 5, 6])

print(P)
print(T)

# extract the coordinates
x = P[:, 0]
y = P[:, 1]
z = P[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, y, z, c=T, cmap='viridis')
for i in range(len(T)):
    ax.text(x[i], y[i], z[i], str(T[i]))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()