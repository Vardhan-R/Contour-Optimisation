from queue import PriorityQueue
import matplotlib.pyplot as plt, numpy as np

SQRT_2 = np.sqrt(2)

# v = lambda x, y: np.exp(-(np.sqrt(x ** 2 + y ** 2) - 1) ** 2)
# v = lambda x, y: 2 + np.sin(2 * x) + np.cos(2 * y)
# v = lambda x, y: np.abs(2 + np.sin(2 * x) + np.exp(y / 2))
# v = lambda x, y: np.abs(x * y ** 2 / (y + 5) + 1.8)
# v = lambda x, y: 1.0 + x - x + y - y
# v = lambda x, y: 1 / (1.1 + 0.1 * np.sin(np.pi * x))
v = lambda x, y: 1 / (1.3 + 0.1 * np.sin(np.pi * x) + 0.2 * np.sin(2 * np.pi * x + np.pi / 4))

# x_i, y_i = -2.5, 1.1
# x_f, y_f = 2, -0.3
# x_i, y_i = 2, 0
# x_f, y_f = 0, 2
# x_i, y_i = 3, 3
# x_f, y_f = -3, -1
# x_i, y_i = -2.5, 1.1
# x_f, y_f = -2.4, 1.5
x_i, y_i = 0, 0
x_f, y_f = 10, 20

l = 20
n = 101
x_range = np.linspace(-l, l, n)
y_range = np.linspace(-l, l, n)
xx, yy = np.meshgrid(x_range, y_range)
z = v(xx, yy)
# dx = dy = 2 * l / (n - 1)
ds = 2 * l / (n - 1)

coord_to_ind = lambda x, y: (round((y + l) * (n - 1) / (2 * l)), round((x + l) * (n - 1) / (2 * l)))
cost = lambda x, y: ds / z[round((y + l) * (n - 1) / (2 * l)), round((x + l) * (n - 1) / (2 * l))]
i_i, j_i = coord_to_ind(x_i, y_i)
i_f, j_f = coord_to_ind(x_f, y_f)

# steps = 101
# step_size = np.sqrt((x_f - x_i) ** 2 + (y_f - y_i) ** 2) / steps

def fourWaySearch():
	# check right
	try:
		x_next, y_next = x_curr + ds, y_curr
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next < n and j_next < n:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check left
	try:
		x_next, y_next = x_curr - ds, y_curr
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next >= 0 and j_next >= 0:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check up
	try:
		x_next, y_next = x_curr, y_curr + ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next >= 0 and j_next >= 0:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check down
	try:
		x_next, y_next = x_curr, y_curr - ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next < n and j_next < n:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

def eightWaySearch():
	# check right
	try:
		x_next, y_next = x_curr + ds, y_curr
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next < n and j_next < n:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check left
	try:
		x_next, y_next = x_curr - ds, y_curr
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next >= 0 and j_next >= 0:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check up
	try:
		x_next, y_next = x_curr, y_curr + ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next >= 0 and j_next >= 0:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check down
	try:
		x_next, y_next = x_curr, y_curr - ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next < n and j_next < n:
			q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check top-right
	try:
		x_next, y_next = x_curr + ds, y_curr + ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next < n and j_next < n:
			q.put((curr_cost + SQRT_2 * cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check top-left
	try:
		x_next, y_next = x_curr - ds, y_curr + ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next >= 0 and j_next >= 0:
			q.put((curr_cost + SQRT_2 * cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check bottom-right
	try:
		x_next, y_next = x_curr + ds, y_curr - ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next >= 0 and j_next >= 0:
			q.put((curr_cost + SQRT_2 * cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

	# check bottom-left
	try:
		x_next, y_next = x_curr - ds, y_curr - ds
		i_next, j_next = coord_to_ind(x_next, y_next)
		if not visited[i_next][j_next] and i_next < n and j_next < n:
			q.put((curr_cost + SQRT_2 * cost(x_next, y_next), x_next, y_next, x_path, y_path))
			visited[i_next][j_next] = True
	except:
		pass

def radialSearch(angs: int):
	for ang in range(angs):
		try:
			x_next, y_next = x_curr + ds * np.cos(2 * np.pi * ang / angs), y_curr + ds * np.sin(2 * np.pi * ang / angs)
			i_next, j_next = coord_to_ind(x_next, y_next)
			if not visited[i_next][j_next] and i_next < n and j_next < n:
			# if 0 <= i_next < n and 0 <= j_next < n:
				q.put((curr_cost + cost(x_next, y_next), x_next, y_next, x_path, y_path))
				visited[i_next][j_next] = True
		except:
			pass

# BFS with keeping track of the cumulative cost (sum(ds / v(x_curr, y_curr)))
q = PriorityQueue()
# x_curr, y_curr = x_i, y_i
visited = [[False] * n for _ in range(n)]
q.put((cost(x_i, y_i), x_i, y_i, [], [])) # (cumulative cost, x_curr, y_curr, x_path, y_path)
# i_curr, j_curr = coord_to_ind(x_curr, y_curr)
visited[i_i][j_i] = True

cnt = 0
while not q.empty():
	# get best dir
	curr_cost, x_curr, y_curr, x_path, y_path = q.get_nowait()
	i_curr, j_curr = coord_to_ind(x_curr, y_curr)
	# visited[i_curr][j_curr] = True
	cnt += 1
	if not cnt % 1000:
		print(cnt, i_curr, j_curr, curr_cost)
		# print(q.qsize())

	x_path = x_path + [x_curr]
	y_path = y_path + [y_curr]

	# break if the end has been reached
	if i_curr == i_f and j_curr == j_f:
		print(cnt, i_curr, j_curr, curr_cost)
		break

	# fourWaySearch()
	# eightWaySearch()
	radialSearch(10)

print(i_i, j_i)
print(i_f, j_f)
plt.contourf(xx, yy, z, levels=np.linspace(0, np.max(z), n), cmap="magma")
plt.colorbar()
plt.plot(x_path, y_path)
plt.scatter((x_i, x_f), (y_i, y_f))
plt.axis("scaled")
plt.show()
