
import pickle
import operator


def plot(model):
    from xgboost import plot_importance
    from matplotlib import pyplot as plt
    plot_importance(model)
    plt.show()


if __name__ == '__main__':

    filename = 'model/xgb_depth_7_round_1800_fold_2_eta_0.002.pkl'

    model = pickle.load(open(filename))

    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    importance = importance[::-1]
    print(importance)
    plot(model)


