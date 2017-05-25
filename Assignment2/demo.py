def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    mu1 = 1
    mu2 = 1
    mu3 = 1
    mu4 = 1
    mu5 = 1
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    c5 = 0;
    mean1 = 0
    mean2 = 0

    # print(y)
    for i in range(len(X)):
        if (y[i] == 1):
            mu1 += X[i]
            c1 += 1
            # print(mu1)
            # print(sum(X[i]))
        elif (y[i] == 2):
            mu2 += (X[i])
            c2 += 1
            # print(mu2)
        elif (y[i] == 3):
            mu3 += (X[i])
            c3 += 1
            # print(mu3)
        elif (y[i] == 4):
            mu4 += (X[i])
            c4 += 1
            # print(mu4)
        elif (y[i] == 5):
            mu5 += (X[i])
            c5 += 1
            # print(mu5, c5)

    mean1 += sum(X[:, 0])
    mean2 += sum(X[:, 1])

    mu1 /= c1;
    mu2 /= c2;
    mu3 /= c3;
    mu4 /= c4;
    mu5 /= c5;

    np.append([mu1], [mu2], axis=0)
    means = np.array([mu1, mu2, mu3, mu4, mu5])

    # print(mu1, mu2,mu3,mu4,mu5)
    means = means.transpose()
    # means=np.concatenate((mu1, mu2,mu3,mu4,mu5));
    # print(means)

    mean1 /= len(X);
    mean2 /= len(X);
    mean = np.array([mean1, mean2])
    # print("mu",mean)


    covmat = np.dot((X - mean).transpose(), (X - mean)) * (1 / len(X))
    # print(len(X))
    # mat = sum(np.power((X-mean), 2)) * (1/len(X))
    # print("Covmat",covmat)

    # covmat = np.array([[mat[0],0],[0,mat[1]]])

    # print(means)

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    mu1 = 1
    mu2 = 1
    mu3 = 1
    mu4 = 1
    mu5 = 1
    c1 = 0;
    c2 = 0;
    c3 = 0;
    c4 = 0;
    c5 = 0;
    mean1 = 0
    mean2 = 0

    # print(y)
    for i in range(len(X)):
        if (y[i] == 1):
            mu1 += X[i]
            c1 += 1

        elif (y[i] == 2):
            mu2 += (X[i])
            c2 += 1

        elif (y[i] == 3):
            mu3 += (X[i])
            c3 += 1

        elif (y[i] == 4):
            mu4 += (X[i])
            c4 += 1

        elif (y[i] == 5):
            mu5 += (X[i])
            c5 += 1

    mean1 += sum(X[:, 0])
    mean2 += sum(X[:, 1])

    mu1 /= c1
    mu2 /= c2
    mu3 /= c3
    mu4 /= c4
    mu5 /= c5

    means = np.array([mu1, mu2, mu3, mu4, mu5])
    means = means.transpose()

    covmats = []
    for i in range(len(np.unique(ytest))):
        select_indices = np.where(y == (i + 1))[0]
        # print(X[select_indices])
        covmat = np.dot((X[select_indices] - means[:, i]).transpose(), (X[select_indices] - means[:, i])) * (
        1 / len(select_indices))
        covmats.append(covmat)

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    # np.exp(-1*(Xtest[]))

    list = []

    for i in range(len(np.unique(ytest))):
        mat = np.dot((Xtest - means[:, i]), np.linalg.inv(covmat))
        mat = mat * (Xtest - means[:, i])
        result = np.exp(-1 * np.sum(mat, axis=1))
        list.append(result)

    ypred = np.array(list).transpose()
    ypred = ypred.argmax(1) + 1
    ypred = np.reshape(ypred, (len(Xtest), 1))

    acc = np.sum(ypred == ytest) * (1 / len(Xtest))

    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    list = []

    for i in range(len(np.unique(ytest))):
        mat = np.dot((Xtest - means[:, i]), np.linalg.inv(covmats[i]))
        mat = mat * (Xtest - means[:, i])
        result = np.exp(-1 * np.sum(mat, axis=1))
        list.append(result)

    ypred = np.array(list).transpose()
    ypred = ypred.argmax(1) + 1
    ypred = np.reshape(ypred, (len(Xtest), 1))

    acc = np.sum(ypred == ytest) * (1 / len(Xtest))

    return acc, ypred