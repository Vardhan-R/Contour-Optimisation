import matplotlib.pyplot as plt, numpy as np

def refractiveIndex(x: float | int, y: float | int = 0) -> float | int:
	return 1.3 + 0.1 * np.sin(np.pi * x) + 0.2 * np.sin(2 * np.pi * x + np.pi / 4)

x_i, y_i = 0, 0
x_f, y_f = 10, 20
delta_x, delta_y = 1e-5, 1e-5
ang_i = np.pi / 4 - 0.1

x = np.arange(x_i, x_f + delta_x, delta_x)
# y = np.arange(y_i, y_f + delta_y, delta_y)
y = np.linspace(y_i, y_f, 2)
xx, yy = np.meshgrid(x, y)
ref_ind = refractiveIndex(xx, yy)

all_x, all_y = [x_i], [y_i]

denom = (refractiveIndex(x_i) * np.sin(ang_i)) ** 2

for k in range(1, int((x_f - x_i) / delta_x + 1)):
	all_x.append(all_x[-1] + delta_x)
	all_y.append(all_y[-1] + delta_x / np.sqrt(refractiveIndex(x_i + k * delta_x) ** 2 / denom - 1))

plt.contourf(xx, yy, ref_ind, levels=np.linspace(1, 1.6, 51), cmap="Blues")
plt.colorbar()
# plt.scatter((x_i, x_f), (y_i, y_f), color='g')
plt.plot(all_x, all_y, color='r')
plt.show()
