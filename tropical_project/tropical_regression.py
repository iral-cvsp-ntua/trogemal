import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans


def get_data(file_name):
    read_data = genfromtxt(file_name)
    data_x = read_data[:, 0]
    n = np.shape(data_x)[0]
    data_x = np.reshape(data_x, (n,))
    data_y = read_data[:, 1]
    data_y = np.reshape(data_y, (n,))
    target = read_data[:, 2]
    target = np.reshape(target, (n,))
    return data_x, data_y, target


def norm_2(y, target):
    return np.sqrt(np.mean((y - target) ** 2))


def norm_inf(y, target):
    error = abs(y[0] - target[0])
    for idx in range(len(y)):
        value = abs(y[idx] - target[idx])
        if value > error:
            error = value
    return error


def calculate_output(data_x, data_y, target, centers, w):
    k = centers.shape[0]
    shape = data_x.shape
    z = -1 * np.ones(shape)
    for idx in range(k):
        z = np.maximum(z, centers[idx][0] * data_x + centers[idx][1] * data_y + w[idx])
    return z


def print_statistics(data_x, data_y, target, centers, w):
    z = calculate_output(data_x, data_y, target, centers, w)
    rms = norm_2(z, target)
    linf = norm_inf(z, target)
    print("The errors for the plane are:")
    print("L2: ", rms)
    print("L_inf: ", linf)
    print(" ")
    print("OPTIMUM:")
    mu = linf / 2
    z = calculate_output(data_x, data_y, target, centers, w + mu)
    rms = norm_2(z, target)
    linf = norm_inf(z, target)
    print("L2: ", rms)
    print("L_inf: ", linf)
    return 0


def plot_graph(data_x, data_y, target, centers, w):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xv, yv = np.meshgrid(x, y)

    z = calculate_output(xv, yv, target, centers, w)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data_x, data_y, target, c="k", marker=".")

    # Normalize the colormap
    norm = plt.Normalize(z.min(), z.max())
    colors = cm.viridis(norm(z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(
        xv, yv, z, rcount=rcount / 3, ccount=ccount / 3, facecolors=colors
    )
    surf.set_facecolor((0, 0, 0, 0))
    ax.xaxis.set_ticks(np.arange(-10, 10.1, 2))
    ax.yaxis.set_ticks(np.arange(-10, 10.1, 2))
    # plt.savefig("graph{}.pdf".format(np.random.randint(10)))
    plt.show()
    text_file = open("tropic_data" + str(np.random.randint(10)) + ".txt", "w")
    for idx in range(100):
        for jdx in range(100):
            text_file.write("%s " % xv[idx][jdx])
            text_file.write("%s " % yv[idx][jdx])
            text_file.write("%s\n" % z[idx][jdx])
    text_file.close()
    return 0


def solve_max2d(data, target):
    b = []
    b.append(np.min(target - data[:][0]))
    b.append(np.min(target - data[:][1]))
    b.append(np.min(target))
    return b


def solve_max2d_quad(data, target):
    b = []
    b.append(np.min(target - 2 * data[:][1]))
    b.append(np.min(target - 2 * data[:][0]))
    b.append(np.min(target - data[:][1] - data[:][0]))
    b.append(np.min(target - data[:][1]))
    b.append(np.min(target - data[:][0]))
    b.append(np.min(target))
    b.append(np.min(target + data[:][0]))
    b.append(np.min(target + data[:][1]))
    b.append(np.min(target + data[:][1] + data[:][0]))
    b.append(np.min(target + 2 * data[:][0]))
    b.append(np.min(target + 2 * data[:][1]))
    return b


def solve_max2d_variable(data, target, k):
    shape = data[:][0].shape
    grads = np.gradient(data)
    grads_x = grads[0]
    grads_x = grads_x[:][0]
    grads_x = np.reshape(grads_x, shape)

    grads_y = np.array(grads[1])
    grads_y = grads_y[:][0]
    grads_y = np.reshape(grads_y, shape)

    grads = np.array([grads_x, grads_y])
    # Kmeans expects data in a different convention
    shape = data.shape
    shape = (shape[1], shape[0])
    grads = np.reshape(grads, shape)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(grads)
    centers = kmeans.cluster_centers_

    b = []
    for idx in range(k):
        b.append(
            np.min(target - centers[idx][0] * data[:][0] - centers[idx][1] * data[:][1])
        )
    return b, centers


data_x = []
data_y = []
target = []

m = 100
for idx in range(m):
    x = 10 * np.random.random()
    y = 10 * np.random.random()
    z = max(x + 5, 9)
    z = max(z, y + 7) + np.random.randn()

    data_x.append(x)
    data_y.append(y)
    target.append(z)

data_x = np.array(data_x)
data_x = np.reshape(data_x, (m,))
data_y = np.array(data_y)
data_y = np.reshape(data_y, (m,))
target = np.array(target)
target = np.reshape(target, (m,))
# data_x, data_y, target = get_data("data.txt")
data = np.array([data_x, data_y])

w_1 = solve_max2d(data, target)
print(w_1)
w_2 = solve_max2d_quad(data, target)
k = 10
w_3, centers = solve_max2d_variable(data, target, k)


# Plot plane
centers_plane = np.array([[1, 0], [0, 1], [0, 0]])
print_statistics(data_x, data_y, target, centers_plane, w_1)
print("----------------------------------------")
plot_graph(data_x, data_y, target, centers_plane, w_1)
z = calculate_output(data_x, data_y, target, centers_plane, w_1)
linf = norm_inf(z, target)
mu = linf / 2
plot_graph(data_x, data_y, target, centers_plane, w_1 + mu)

# # Plot 2d
# centers_2d = np.array([[0, 2], [2, 0], [1, 1], [0, 1], [1, 0], [0, 0], [-1, 0], [0, -1], [-1, -1], [-2, 0], [0, -2]])
# print_statistics(data_x, data_y, target, centers_2d, w_2)
# print("----------------------------------------")
# plot_graph(data_x, data_y, target, centers_2d, w_2)
# z = calculate_output(data_x, data_y, target, centers_2d, w_2)
# linf = norm_inf(z, target)
# mu = linf / 2
# plot_graph(data_x, data_y, target, centers_2d, w_2 + mu)
#
# # Plot variable
# print_statistics(data_x, data_y, target, centers, w_3)
# # plot_graph(data_x, data_y, target, centers, w_3)
# z = calculate_output(data_x, data_y, target, centers, w_3)
# linf = norm_inf(z, target)
# mu = linf / 2
# plot_graph(data_x, data_y, target, centers, w_3 + mu)

text_file = open("data.txt", "w")
for idx in range(len(target)):
    text_file.write("%s " % data[0][idx])
    text_file.write("%s " % data[1][idx])
    text_file.write("%s\n" % target[idx])
text_file.close()
