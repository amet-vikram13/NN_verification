import gurobipy as gp
import numpy as np
from gurobipy import GRB
from tqdm import tqdm
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from tensorflow.keras.datasets import mnist

data_path = "/data/local/AA/data/"
results_path = "/data/local/AA/results/"
coresets_path = "/data/local/AA/results/coresets/"

# renaming some data sets
data_name = { # old:new
    "ijcnn1": "Ijcnn1",
    "pose": "Pose",
    "song": "Song",
    "covertype": "Covertype",
}

def load_data(dataset, data=None, standardize=False):
    X = []
    y = []

    if dataset == "covertype":  # (581012, 54)
        # Forest cover type
        # https://archive.ics.uci.edu/ml/datasets/covertype
        X, y = load_svmlight_file(data_path + "covtype.libsvm.binary")
        X = np.asarray(X.todense())
    elif dataset == "ijcnn1":  # (49990, 22)
        # https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        X, y = load_svmlight_file(data_path + "ijcnn1.bz2")
        X = np.asarray(X.todense())
    elif dataset == "song":  # (515345, 90)
        # YearPredictionMSD is a subset of the Million Song Dataset
        # https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
        data = np.loadtxt(
            data_path + "YearPredictionMSD.txt", skiprows=0, delimiter=","
        )
        X = data[:, 1:]
        y = data[:, 0]
    elif dataset == "pose":  # (35832, 48)
        # ECCV 2018 PoseTrack Challenge
        # http://vision.imar.ro/human3.6m/challenge_open.php
        X = []
        for i in tqdm(range(1, 35832 + 1), desc="loading pose"):
            f = data_path + "Human3.6M/ECCV18_Challenge/Train/POSE/{:05d}.csv".format(i)
            data = np.loadtxt(f, skiprows=0, delimiter=",")
            X.append(data[1:, :].flatten())
        X = np.array(X)
    elif data is not None:
        return data, y
    else:
        raise NotImplementedError

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, y

# Determines whether a point s is a convex combination
# of the points in the set E.
# Returns None if s is a convex combination of E,
# otherwise returns a witness vector that certifies
# that s is not a convex combination of E.
def isConvexCombination(X, ind_E, s):
    E = X[ind_E].copy()
    P = X[s].copy()

    # initialize the dimensions of the data
    k = E.shape[0]
    d = E.shape[1]

    # initialize the model and parameters
    model = gp.Model("ConvexCombination")
    model.setParam("OutputFlag", 0)

    # adding model variables -- the lambda coefficients of the convex combination equation should be between 0 and 1
    lambdas = model.addMVar((k,), lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="lambdas")

    # adding model constraints -- the sum of the lambda coefficients should be 1
    model.addConstr(lambdas.sum() == 1, name="sum_of_lambdas")

    # adding model constraints -- the convex combination equation => E.T x lambdas = P
    model.addMConstr(E.T, lambdas, '=', P, "convex_combination_equation")

    model.setParam("InfUnbdInfo", 1)

    # optimize the model
    # model.write("_gurobi_lp/convex_combination.lp")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return None
    else:
        # model.computeIIS()
        # model.write("_gurobi_lp/convex_combination.ilp")

        # Computing the Farkas dual, when model is infeasible.
        # The Farkas dual is a certificate of infeasibility.
        # It is a vector that satisfies the following conditions:
        # y.T * A * x <= y.T * b
        # when the original problem : A * x = b is infeasible.
        # Here y will be our witness vector.
        dual = []
        for i in range(d):
            constr = model.getConstrByName("convex_combination_equation[{}]".format(i))
            assert constr is not None
            dual.append(constr.getAttr(GRB.Attr.FarkasDual))
        return np.array(dual)

# finds set of points that are farthest apart
# using simple min max along each dimension of X
def farthestPointsSetUsingMinMax(X):
    n = X.shape[0]
    d = X.shape[1]

    ind_E = set()

    for i in range(d):
        p1 = X[:,i].argmin()
        p2 = X[:,i].argmax()
        ind_E.add(p1)
        ind_E.add(p2)

    return list(ind_E)

# proposed coreset
# "clarkson-cs" in the paper "More output-sensitive geometric algorithms"
def clarkson_coreset(X, ind_E, ind_S, dataset_name):
    X_C = np.empty((1, 1))
    try:
        data = np.load(coresets_path + dataset_name + "_clarkson_coreset.npz")
        X_C = data["X"]
    except FileNotFoundError:
        t_start = time()
        try:
            pbar = tqdm(total=len(ind_S), desc="clarkson-cs computation:")
            while len(ind_S) > 0:
                if len(ind_E) % 1000 == 0:
                    pbar.write(
                        "Current Size of coreset: {}".format(len(ind_E)))
                    pbar.write(
                        "Remaining points to process:: {}".format(len(ind_S)))
                s = ind_S.pop(0)
                witness_vector = isConvexCombination(X, ind_E, s)
                if witness_vector is not None:
                    max_dot_product = np.dot(-1*witness_vector, X[s])
                    p_prime = None
                    for p in ind_S:
                        dot_product = np.dot(-1*witness_vector, X[p])
                        if dot_product > max_dot_product:
                            max_dot_product = dot_product
                            p_prime = p
                    if p_prime is not None:
                        ind_E.append(p_prime)
                        ind_S.append(s)
                        ind_S.remove(p_prime)
                    else:
                        ind_E.append(s)
                pbar.update(1)
            pbar.close()
        except Exception as e:
            print(e)
        X_C = X[ind_E].copy()
        t_end = time()
        np.savez(
            coresets_path + dataset_name + "_clarkson_coreset.npz",
            X=X_C,
            cs_time=t_end - t_start
        )
    finally:
        return X_C

def compute_clarkson_coreset(dataset, data=None):
    X, _ = load_data(dataset, data=data)  # y won't be used

    t_start = time()

    # initialize two extreme points via farthestPointsSetUsingMinMax function
    # maintain the initialized indices as set E. Note: len(E) < len(X)
    # maintain the indices not belonging to E as set S. Note: len(S) = len(X) - len(E)
    # any index not belonging to E is a candidate for the next coreset
    ind_E = farthestPointsSetUsingMinMax(X)
    ind_S = np.setdiff1d(np.arange(len(X)), np.array(ind_E)).tolist()

    # obtain initial coreset using Clarkson's algorithm
    X_C = clarkson_coreset(X, ind_E, ind_S, dataset)

    t_end = time()

    print("Length of X_C: ", len(X_C))
    print("Time taken: ", t_end-t_start)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate((x_train, x_test))
    X = X.reshape(X.shape[0], 28 * 28)
    compute_clarkson_coreset("mnist", data=X)
