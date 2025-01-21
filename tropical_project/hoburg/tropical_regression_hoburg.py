import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
#from collections import Counter
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

def solve_max(a, x, f):
	b = []
	for alpha in a:
		b.append(np.min(f - alpha * x))
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

data = np.linspace(-2, 2, 100)
n = np.shape(data)[0]
data = np.reshape(data, (n, 1))

left_part = -6 * data - 6
middle_part = 1/float(2) * data
right_part = 1/float(5) * data ** 5 + 1/float(2) * data
orig = np.maximum(left_part, middle_part)
orig = np.maximum(orig, right_part)

#w_0x, w_1x = tropical_regression_max(data, orig)
#print("The tropical regression (max) coefficients are: " + str(w_0x) + ", " + str(w_1x))

derivs = []
for idx in range(len(data) - 1):
	deriv = (orig[idx+1] - orig[idx]) / float(data[idx+1] - data[idx])
	derivs.append(float(deriv))

# kmeans = KMeans(n_clusters=2, random_state=0).fit(derivs)
# print(kmeans.cluster_centers_)

## HISTOGRAM
hist = np.histogram(derivs, bins=3)
indexes = np.argsort(hist[0])
counts = hist[0]
vals = hist[1]
reverse = indexes[::-1]
counts = counts[reverse]
vals = vals[reverse]
print(derivs)
print("Counts and vals:")
print(counts)
print(vals)

chosen = sorted([float(i) for i in vals[0:4]])
bin1 = []
bin2 = []
bin3 = []
for idx in range(len(derivs)):
	if derivs[idx] <= chosen[0]:
		bin1.append(derivs[idx])
	elif derivs[idx] <= chosen[1]:
		bin2.append(derivs[idx])
	elif derivs[idx] <= chosen[2]:
		bin3.append(derivs[idx])

val1 = np.mean(bin1)
val2 = np.mean(bin2)
val3 = np.mean(bin3)
print(val1, val2, val3)

plt.hist(derivs, bins='auto')
plt.show()

a = [float(i) for i in vals[0:3]]
a = [0.5, -6, 3]
#a = [val1, val2, val3]
#a.append(0)
print("Alphas")
print(a)
w = solve_max(a, data, orig)
print("The coefficients are", w)

error = linf(a, data, orig, w)
print(error)

y_x = calculate_output(a, w, data)

# MMAE
y_linf = y_x + float(norm_inf(y_x, orig)) / 2

print("Errors for the GLE:")
print("L2: " + str(norm_2(y_x, orig)))
print("L_inf: " + str(norm_inf(y_x, orig)[0]))

print("Errors for the MMAE:")
print("L2: " + str(norm_2(y_linf, orig)))
print("L_inf: " + str(norm_inf(y_linf, orig)[0]))

plt.plot(data, orig, '.k', fillstyle='none')
plt.plot(data, y_x, 'r', label="Tropical line")
plt.plot(data, y_linf, 'g', label="Tropical linf")
plt.xlim([-2, 2])
plt.ylim([-2, 8])
plt.xlabel("Data (x)")
plt.ylabel("Target")
plt.title("Max-plus tropical line fitting")
plt.legend()
plt.show()

# Write to file
text_file = open("tropical_max.txt", "w")
for idx in range(n):
    text_file.write("%s " % data[idx][0])
    text_file.write("%s\n" % y_x[idx][0])
text_file.close()

text_file = open("tropical_max_linf.txt", "w")
for idx in range(n):
    text_file.write("%s " % data[idx][0])
    text_file.write("%s\n" % y_linf[idx][0])
text_file.close()

text_file = open("data.txt", "w")
for idx in range(n):
    text_file.write("%s " % data[idx][0])
    text_file.write("%s\n" % orig[idx][0])
text_file.close()
