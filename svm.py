import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


class SMOStruct:
    def __init__(self, X, y, C, kernel, toler):
        self.N = X.shape[0]
        self.X = X  # sample
        self.y = y  # label
        self.C = C  # regularization parameter
        self.kernel = kernel  # kernel function
        self.b = 0  # scalar bias term
        self.E = np.zeros(self.N)
        self.toler = toler
        self.lambdas = np.zeros(self.N)  # lagrange multiplier
        self.K = np.zeros((self.N, self.N))  # kernel value
        for i in range(self.N):
            for j in range(self.N):
                self.K[i, j] = Kernel(X[i].T, X[j].T, kernel)


# load data
def load_data(path, flag):
    data = pd.read_csv(path, sep='\t', names=[i for i in range(3)])
    data = np.array(data).tolist()
    x = []
    y = []
    for i in range(len(data)):
        y.append(data[i][-1])
        del data[i][-1]
        x.append(data[i])

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    x = np.array(x)
    y = np.array(y).reshape(len(y), )

    train_x = x[0:80]
    train_y = y[0:80]
    test_x = x[80:len(x)]
    test_y = y[80:len(x)]

    if flag:
        return x, y
    return x, y, train_x, train_y, test_x, test_y


# calculate kernel value
def Kernel(xi, xj, kernel):
    res = 0.0
    if kernel[0] == 'linear':  # linear kernel
        res = np.dot(xi.T, xj)
    elif kernel[0] == 'rbf':  # Gaussian kernel
        sum = 0.0
        for i in range(len(xi)):
            sum += (xi[i] - xj[i]) ** 2
        res = np.exp(-sum / (2 * kernel[1] ** 2))
    elif kernel[0] == 'poly':  # poly kernel
        res = np.dot(xi.T, xj)
        res = res ** kernel[1]
    elif kernel[0] == 'laplace':  # Laplace kernel
        sum = 0.0
        for i in range(len(xi)):
            sum += (xi[i] - xj[i]) ** 2
            sum = sum ** 0.5
        res = np.exp(-sum / kernel[1])
    elif kernel[0] == 'sigmoid':  # Sigmoid kernel
        res = np.dot(xi.T, xj)
        res = np.tanh(kernel[1] * res + kernel[2])

    return res


# Violation of KKT condition
def kkt(model, i):
    Ei = cacEi(model, i)
    if ((model.y[i] * Ei < -model.toler) and (model.lambdas[i] < model.C)) or (
            (model.y[i] * Ei > model.toler) and (model.lambdas[i] > 0)):
        return True
    else:
        return False


# choose lambda_i
def outerLoop(model):
    for i in range(model.N):
        if kkt(model, i):
            return i


# choose lambda_j
def inner_loop(model, i):
    E1 = cacEi(model, i)
    max_diff = 0
    j = None
    E = np.nonzero(model.E)[0]
    # print(model.E)
    if len(E) > 1:
        for k in E:
            if k == i:
                continue
            Ek = cacEi(model, k)
            diff = np.abs(Ek - E1)
            # print("diff",diff)
            if diff > max_diff:
                max_diff = diff
                j = k
    else:
        j = i
        while j == i:
            j = int(np.random.uniform(0, model.N))
    return j


# calculate f(xk)
def calfx(model, k):
    sum = 0.0
    for i in range(model.N):
        sum += (model.lambdas[i] * model.y[i] * model.K[i, k])
    sum += model.b
    return sum


# calculate Ek
def cacEi(model, k):
    return calfx(model, k) - float(model.y[k])


# core program
def update(model, i, j):
    # print(i, j)
    lambdai_old = model.lambdas[i]
    lambdaj_old = model.lambdas[j]
    if model.y[i] != model.y[j]:
        L = max(0.0, model.lambdas[j] - model.lambdas[i])
        H = min(model.C, model.C + model.lambdas[j] - model.lambdas[i])
    else:
        L = max(0.0, model.lambdas[j] + model.lambdas[i] - model.C)
        H = min(model.C, model.lambdas[j] + model.lambdas[i])
    if L == H:
        return 0
    eta = model.K[i, i] + model.K[j, j] - 2.0 * model.K[i, j]
    if eta <= 0:
        return 0
    lambdaj_new_unc = lambdaj_old + model.y[j] * (model.E[i] - model.E[j]) / eta
    lambdaj_new = clipBonder(L, H, lambdaj_new_unc)
    lambdai_new = lambdai_old + model.y[i] * model.y[j] * (lambdaj_old - lambdaj_new)
    # update lambda
    model.lambdas[i] = lambdai_new
    model.lambdas[j] = lambdaj_new

    # update b
    b1 = model.b - model.E[i] - model.y[i] * (lambdai_new - lambdai_old) * model.K[i, i] - model.y[j] * (
            lambdaj_new - lambdaj_old) * model.K[i, j]
    b2 = model.b - model.E[j] - model.y[i] * (lambdai_new - lambdai_old) * model.K[i, j] - model.y[j] * (
            lambdaj_new - lambdaj_old) * model.K[j, j]

    if 0 < model.lambdas[i] < model.C:
        model.b = b1
    elif 0 < model.lambdas[j] < model.C:
        model.b = b2
    else:
        model.b = (b1 + b2) / 2.0
    # update E
    model.E[i] = cacEi(model, i)
    model.E[j] = cacEi(model, j)

    return 1


# calculate w
def calW(lambdas, X, y):
    m, n = X.shape
    w = np.zeros((n, 1))
    for i in range(n):
        for j in range(m):
            w[i] += (lambdas[j] * y[j] * X[j, i])
    return w


# clip
def clipBonder(L, H, lambda_):
    if lambda_ > H:
        return H
    elif H >= lambda_ >= L:
        return lambda_
    else:
        return L


# smo main program
def smo_main(C, kernel, toler):
    x, y, train_x, train_y, test_x, test_y = load_data('svm_data/svmDataSet.txt', False)
    # train_x, train_y = load_data('horse_colic/horseColicTraining.txt', True)
    model = SMOStruct(train_x, train_y, C, kernel, toler)
    max_step = 50
    while max_step > 0:
        for i in range(model.N):
            if kkt(model, i):
                j = inner_loop(model, i)
                update(model, i, j)
        max_step -= 1

    # entireSet = True
    # lambdaPairsChanged = 0
    #
    # while(max_step > 0) and (lambdaPairsChanged > 0 or entireSet):
    #     lambdaPairsChanged = 0
    #     if entireSet:
    #         for i in range(model.N):
    #             if kkt(model, i):
    #                 j = inner_loop(model, i)
    #                 lambdaPairsChanged += update(model, i, j)
    #                 max_step -= 1
    #     else:
    #         nonBoundLambdaList = np.nonzero(model.lambdas != 0 and model.lambdas != model.C)[0]
    #         for i in nonBoundLambdaList:
    #             if kkt(model, i):
    #                 j = inner_loop(model, i)
    #                 lambdaPairsChanged += update(model, i, j)
    #                 max_step -= 1
    #     if entireSet:
    #         entireSet = False
    #     elif lambdaPairsChanged == 0:
    #         entireSet = True

    w = calW(model.lambdas, model.X, model.y)
    return model, model.lambdas, w, model.b


# plot
def plotSVM(model, w):
    x, y, train_x, train_y, test_x, test_y = load_data('svm_data/svmDataSet.txt', False)
    w1 = w[0, 0]
    w2 = w[1, 0]
    b = model.b
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(len(x)):
        res = np.dot(w.T, x[i, :]) + b
        res = np.sign(res)
        if res == 1:
            x0.append(x[i, 0])
            y0.append(x[i, 1])
        else:
            x1.append(x[i, 0])
            y1.append(x[i, 1])
    plt.scatter(x0, y0, c='red', marker='*')
    plt.scatter(x1, y1, c='blue', marker='*')
    plt.legend(['Class_1', 'Class_2'])
    x2 = np.linspace(0, 10, 1000)
    y2 = -b / w2 - w1 / w2 * x2
    plt.plot(x2, y2)
    plt.show()


def SVM():
    C = 1.0
    toler = 0.001
    kernel = ['linear', 0.5, 0.0]
    model, lambdas, w, b = smo_main(C, kernel, toler)
    x, y, train_x, train_y, test_x, test_y = load_data('svm_data/svmDataSet.txt', False)
    # train_x, train_y = load_data('horse_colic/horseColicTraining.txt', True)
    # test_x, test_y = load_data('horse_colic/horseColicTest.txt', True)
    sum = 0
    for i in range(len(test_y)):
        res = np.dot(w.T, test_x[i, :]) + b
        res = np.sign(res)
        if res == test_y[i]:
            sum += 1

    print('手写正确率：%.2f%%' % (sum / len(test_y) * 100))
    plotSVM(model, w)


def sklearn_svm():
    x, y, train_x, train_y, test_x, test_y = load_data('svm_data/svmDataSet.txt', False)
    # print(train_x)
    # print(train_y)
    # train_x, train_y = load_data('horse_colic/horseColicTraining.txt', True)
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    print('调包正确率：%.2f%%' % (clf.score(test_x, test_y) * 100))


if __name__ == '__main__':
    SVM()
    sklearn_svm()
