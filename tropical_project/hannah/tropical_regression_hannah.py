import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

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

read_data = genfromtxt('data.txt')
data_x = read_data[:, 0]
n = np.shape(data_x)[0]
data_x = np.reshape(data_x, (n, 1))
data_y = read_data[:, 1]
data_y = np.reshape(data_y, (n, 1))
target = read_data[:, 2]
target = np.reshape(target, (n, 1))

data = np.array([data_x, data_y])
w_1 = solve_max2d(data, target)
print(w_1)
w_2 = solve_max2d_quad(data, target)
print(w_2)

# Try meshgrid
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


# Try meshgrid round 2
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xv, yv = np.meshgrid(x, y)

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

#data = (1 - (-1)) * np.random.random((500, 2)) + (-1)
#target = np.ones((500, 1))
#for idx in range(500):
#	target[idx] = data[idx][0] ** 2 + data[idx][1] ** 2 + np.random.normal(0, 0.25)

#print(data)
#print(target)
#data = np.linspace(-2, 2, 100)
#n = np.shape(data)[0]
#data = np.reshape(data, (n, 1))

# left_part = -6 * data - 6
# middle_part = 1/float(2) * data
# right_part = 1/float(5) * data ** 5 + 1/float(2) * data
# orig = np.maximum(left_part, middle_part)
# orig = np.maximum(orig, right_part)
#
# #w_0x, w_1x = tropical_regression_max(data, orig)
# #print("The tropical regression (max) coefficients are: " + str(w_0x) + ", " + str(w_1x))
# k = 1
# w = solve_max(k, data, orig)
# print(w)
#
# a = range(-k, k + 1)
# error = linf(a, data, orig, w)
# print(error)
#
# y_x = calculate_output(a, w, data)
#
# # max-plus line
# #w_0x = w_0x * np.ones((n, 1))
# #y_x = np.maximum(w_1x + data, w_0x)
#
# # MMAE
# y_linf = y_x + float(norm_inf(y_x, orig)) / 2
#
# print("Errors for the GLE:")
# print("L2: " + str(norm_2(y_x, orig)))
# print("L_inf: " + str(norm_inf(y_x, orig)[0]))
#
# print("Errors for the MMAE:")
# print("L2: " + str(norm_2(y_linf, orig)))
# print("L_inf: " + str(norm_inf(y_linf, orig)[0]))
#
# plt.plot(data, orig, '.k', fillstyle='none')
# plt.plot(data, y_x, 'r', label="Tropical line")
# plt.plot(data, y_linf, 'g', label="Tropical linf")
# plt.xlim([-2, 2])
# plt.ylim([-2, 8])
# plt.xlabel("Data (x)")
# plt.ylabel("Target")
# plt.title("Max-plus tropical line fitting")
# plt.legend()
# plt.show()
#
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
