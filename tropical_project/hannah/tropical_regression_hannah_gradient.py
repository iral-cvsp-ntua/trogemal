import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

def linear_regression(data, target):
    n = np.shape(data)[0]
    sum_x_sq = np.sum(data * data)
    sum_xy = np.sum(data * target)
    sum_x = np.sum(data)
    sum_y = np.sum(target)
    denom = sum_x_sq - sum_x * sum_x / n
    w_1 = (sum_xy - sum_x * sum_y / n) / denom
    w_0 = sum_y / n - w_1 * sum_x / n
    return w_0, w_1

def tropical_regression_max(data, target):
    w_1 = np.amin(target - data)
    w_0 = np.amin(target)
    return w_0, w_1

def norm_inf(y, target):
	error = abs(y[0] - target[0])
	for idx in range(len(y)):
		value = abs(y[idx] - target[idx])
		if value > error:
			error = value
	return error

def norm_2(y, target):
	return np.sqrt(np.mean((y - target) ** 2))

def calculate_rms(data_x, data_y, target, centers, w):
	k = centers.shape[0]
	z = -1 * np.ones((500,))
	for idx in range(k):
		z = np.maximum(z, centers[idx][0] * data_x + centers[idx][1] * data_y + w[idx])
	return np.sqrt(np.mean((z - target) ** 2))

def calculate_inf(data_x, data_y, target, centers, w):
	k = centers.shape[0]
	z = -1 * np.ones((500,))
	for idx in range(k):
		z = np.maximum(z, centers[idx][0] * data_x + centers[idx][1] * data_y + w[idx])

	error = abs(z[0] - target[0])
	for idx in range(len(z)):
		value = abs(z[idx] - target[idx])
		if value > error:
			error = value
	return error

def solve_max(degree, x, f):
	b = []
	for a in range(-degree, degree + 1):
		b.append(np.min(f - a * x))
	return b

def linf(a, x, f, w):
	total_error = None

	for idx in range(len(f)):
		current_error = None
		for jdx in range(len(w)):
			current = abs(a[jdx] * x[idx] + w[jdx] - f[idx])
			if current_error == None:
				current_error = current
			elif current_error > current:
				current_error = current
		if total_error == None:
			total_error = current_error
		elif total_error < current_error:
			total_error = current_error
	return total_error

def calculate_output(a, w, data):
	y = -5 * np.ones(np.shape(data))
	for idx in range(len(w)):
		y = np.maximum(y, a[idx] * data + w[idx])
	return y

def solve_max2d(x, f):
	b = []
	b.append(np.min(f - x[:][0]))
	b.append(np.min(f - x[:][1]))
	b.append(np.min(f))
	return b

def solve_max2d_quad(x, f):
	b = []
	b.append(np.min(f - 2 * x[:][1]))
	b.append(np.min(f - 2 * x[:][0]))
	b.append(np.min(f - x[:][1] - x[:][0]))
	b.append(np.min(f - x[:][1]))
	b.append(np.min(f - x[:][0]))
	b.append(np.min(f))
	b.append(np.min(f + x[:][0]))
	b.append(np.min(f + x[:][1]))
	b.append(np.min(f + x[:][1] + x[:][0]))
	b.append(np.min(f + 2 * x[:][0]))
	b.append(np.min(f + 2 * x[:][1]))
	return b

def solve_max2d_variable(x, f, centers):
	b = []
	k = centers.shape[0]
	for idx in range(k):
		b.append(np.min(f - centers[idx][0] * x[:][0] - centers[idx][1] * x[:][1]))
	return b

read_data = genfromtxt('data.txt')
data_x = read_data[:, 0]
n = np.shape(data_x)[0]
data_x = np.reshape(data_x, (n,))
data_y = read_data[:, 1]
data_y = np.reshape(data_y, (n,))
target = read_data[:, 2]
target = np.reshape(target, (n,))

data = np.array([data_x, data_y])
w_1 = solve_max2d(data, target)
#print(w_1)
w_2 = solve_max2d_quad(data, target)
#print(w_2)

grads = np.gradient(data)
grads_x = grads[0]
grads_x = grads_x[:][0]
grads_x = np.reshape(grads_x, (500,))

grads_y = np.array(grads[1])
grads_y = grads_y[:][0]
grads_y = np.reshape(grads_y, (500,))

grads = np.array([grads_x, grads_y])
grads = np.reshape(grads, (500,2))

k = 75
kmeans = KMeans(n_clusters=k, random_state=0).fit(grads)
centers = kmeans.cluster_centers_
w_3 = solve_max2d_variable(data, target, centers)

# Try meshgrid
centers_plane = np.array([[1, 0], [0, 1], [0, 0]])

rms = calculate_rms(data_x, data_y, target, centers_plane, w_1)
linf = calculate_inf(data_x, data_y, target, centers_plane, w_1)
print("The errors for the simple plane are:")
print("GLE L2: ", rms)
print("GLE L_inf: ", linf)
print(" ")
mu = linf / 2
rms = calculate_rms(data_x, data_y, target, centers_plane, w_1 + mu)
linf = calculate_inf(data_x, data_y, target, centers_plane, w_1 + mu)
print("MMAE L2: ", rms)
print("MMAE L_inf: ", linf)
print("----------------------------------------")

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xv, yv = np.meshgrid(x, y)

z = w_1[2] * np.ones((100, 100))
z = np.maximum(z, xv + w_1[0])
z = np.maximum(z, yv + w_1[1])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data_x, data_y, target, c='k', marker='.')
ax.plot_surface(xv, yv, z)#, cmap=cm.viridis)
plt.show()

w_1 = w_1 + mu
z = w_1[2] * np.ones((100, 100))
z = np.maximum(z, xv + w_1[0])
z = np.maximum(z, yv + w_1[1])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data_x, data_y, target, c='k', marker='.')
ax.plot_surface(xv, yv, z)#, cmap=cm.viridis)
plt.show()

# Try meshgrid round 2
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xv, yv = np.meshgrid(x, y)

centers_2d = np.array([[0, 2], [2, 0], [1, 1], [0, 1], [1, 0], [0, 0], [-1, 0], [0, -1], [-1, -1], [-2, 0], [0, -2]])
rms = calculate_rms(data_x, data_y, target, centers_2d, w_2)
linf = calculate_inf(data_x, data_y, target, centers_2d, w_2)
print("The errors for the quadratic equation are:")
print("GLE L2: ", rms)
print("GLE L_inf: ", linf)
print(" ")
mu = linf / 2
rms = calculate_rms(data_x, data_y, target, centers_2d, w_2 + mu)
linf = calculate_inf(data_x, data_y, target, centers_2d, w_2 + mu)
print("MMAE L2: ", rms)
print("MMAE L_inf: ", linf)
print("----------------------------------------")

z = w_2[5] * np.ones((100, 100))
z = np.maximum(z, 2 * yv + w_2[0])
z = np.maximum(z, 2 * xv + w_2[1])
z = np.maximum(z, xv + yv + w_2[2])
z = np.maximum(z, yv + w_2[3])
z = np.maximum(z, xv + w_2[4])
z = np.maximum(z, -xv + w_2[6])
z = np.maximum(z, -yv + w_2[7])
z = np.maximum(z, -xv - yv + w_2[8])
z = np.maximum(z, -2 * xv + w_2[9])
z = np.maximum(z, -2 * yv + w_2[10])
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data_x, data_y, target, c='b', marker='.')

#Viridis shit
norm = plt.Normalize(z.min(), z.max())
colors = cm.plasma(norm(z))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(xv, yv, z, rcount=rcount/3, ccount=ccount/3, facecolors=colors)
surf.set_facecolor((0, 0, 0, 0))
ax.xaxis.set_ticks(np.arange(-1, 1.1, 0.5))
ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
plt.show()

w_2 = w_2 + mu
z = w_2[5] * np.ones((100, 100))
z = np.maximum(z, 2 * yv + w_2[0])
z = np.maximum(z, 2 * xv + w_2[1])
z = np.maximum(z, xv + yv + w_2[2])
z = np.maximum(z, yv + w_2[3])
z = np.maximum(z, xv + w_2[4])
z = np.maximum(z, -xv + w_2[6])
z = np.maximum(z, -yv + w_2[7])
z = np.maximum(z, -xv - yv + w_2[8])
z = np.maximum(z, -2 * xv + w_2[9])
z = np.maximum(z, -2 * yv + w_2[10])
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data_x, data_y, target, c='b', marker='.')

#Viridis shit
norm = plt.Normalize(z.min(), z.max())
colors = cm.plasma(norm(z))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(xv, yv, z, rcount=rcount/3, ccount=ccount/3, facecolors=colors)
surf.set_facecolor((0, 0, 0, 0))
ax.xaxis.set_ticks(np.arange(-1, 1.1, 0.5))
ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
plt.show()

# Try meshgrid round 3
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xv, yv = np.meshgrid(x, y)

rms = calculate_rms(data_x, data_y, target, centers, w_3)
linf = calculate_inf(data_x, data_y, target, centers, w_3)
print("The errors for the K-means are (k=" + str(k) + "):")
print("GLE L2: ", rms)
print("GLE L_inf: ", linf)
print(" ")
mu = linf / 2
rms = calculate_rms(data_x, data_y, target, centers, w_3 + mu)
linf = calculate_inf(data_x, data_y, target, centers, w_3 + mu)
print("MMAE L2: ", rms)
print("MMAE L_inf: ", linf)

z_t = -1 * np.ones((100, 100))
for idx in range(k):
	z_t = np.maximum(z_t, centers[idx][0] * xv + centers[idx][1] * yv + w_3[idx])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data_x, data_y, target, c='b', marker='.')

#Viridis shit
norm = plt.Normalize(z_t.min(), z_t.max())
colors = cm.plasma(norm(z_t))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(xv, yv, z_t, rcount=rcount/3, ccount=ccount/3, facecolors=colors)
surf.set_facecolor((0, 0, 0, 0))
ax.xaxis.set_ticks(np.arange(-1, 1.1, 0.5))
ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
plt.show()

w_3 = w_3 + mu
z_o = -1 * np.ones((100, 100))
for idx in range(k):
	z_o = np.maximum(z_o, centers[idx][0] * xv + centers[idx][1] * yv + w_3[idx])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(data_x, data_y, target, c='b', marker='.')

#Viridis shit
norm = plt.Normalize(z_o.min(), z_o.max())
colors = cm.plasma(norm(z_o))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(xv, yv, z_o, rcount=rcount/3, ccount=ccount/3, facecolors=colors)
surf.set_facecolor((0, 0, 0, 0))
ax.xaxis.set_ticks(np.arange(-1, 1.1, 0.5))
ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
plt.show()
# Write to file
# text_file = open("tropical_max" + str(k) + ".txt", "w")
# for idx in range(n):
#     text_file.write("%s " % data[idx][0])
#     text_file.write("%s\n" % y_x[idx][0])
# text_file.close()
#
# text_file = open("tropical_max_linf" + str(k) + ".txt", "w")
# for idx in range(n):
#     text_file.write("%s " % data[idx][0])
#     text_file.write("%s\n" % y_linf[idx][0])
# text_file.close()

# text_file = open("data.txt", "w")
# for idx in range(len(target)):
# 	text_file.write("%s " % data[idx][0])
# 	text_file.write("%s " % data[idx][1])
# 	text_file.write("%s\n" % target[idx][0])
# text_file.close()
#
# text_file = open("tropic_base" + str(k) + ".txt", "w")
# for idx in range(100):
# 	for jdx in range(100):
# 		text_file.write("%s " % xv[idx][jdx])
# 		text_file.write("%s " % yv[idx][jdx])
# 		text_file.write("%s\n" % z_t[idx][jdx])
# text_file.close()
#
# text_file = open("tropic_optimal" + str(k) + ".txt", "w")
# for idx in range(100):
# 	for jdx in range(100):
# 		text_file.write("%s " % xv[idx][jdx])
# 		text_file.write("%s " % yv[idx][jdx])
# 		text_file.write("%s\n" % z_o[idx][jdx])
# text_file.close()
