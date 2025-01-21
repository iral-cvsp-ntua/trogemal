import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import jenkspy
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

#### JENKS ####
breaks = jenkspy.jenks_breaks(derivs, n_classes=3)
print(breaks)
chosen = breaks
bin1 = []
bin2 = []
bin3 = []
bin4 = []
bin5 = []
bin6 = []
for idx in range(len(derivs)):
	if derivs[idx] < chosen[0]:
		pass
	elif derivs[idx] <= chosen[1]:
		bin1.append(derivs[idx])
	elif derivs[idx] <= chosen[2]:
		bin2.append(derivs[idx])
	elif derivs[idx] <= chosen[3]:
		bin3.append(derivs[idx])
	#elif derivs[idx] <= chosen[4]:
	#	bin4.append(derivs[idx])
	#elif derivs[idx] <= chosen[5]:
	#	bin5.append(derivs[idx])
	#elif derivs[idx] <= chosen[6]:
	#	bin6.append(derivs[idx])

val1 = np.mean(bin1)
val2 = np.mean(bin2)
val3 = np.mean(bin3)
#val4 = np.mean(bin4)
#val5 = np.mean(bin5)
#val6 = np.mean(bin6)

a = [val1, val2, val3]#, val4]#, val5]#, val6]
print("Alphas")
print(a)
w = solve_max(a, data, orig)
print("The coefficients are", w)

error = linf(a, data, orig, w)
print(error)

y_x = calculate_output(a, w, data)

# MMAE
y_linf = y_x + float(norm_inf(y_x, orig)) / 2

print("The MMAE are", [w_i + float(norm_inf(y_x, orig)) / 2 for w_i in w])

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
