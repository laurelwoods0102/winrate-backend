import numpy as np
import pandas as pd


def weight_sort(name, model_type):
    weights = np.load('./trained_model/weights_{0}_{1}.npy'.format(name.replace(' ', '-'), model_type))
    weights = weights.flatten()
    weights = np.delete(weights, (0), axis=0)   # del bias
    df = pd.DataFrame(weights, columns=["weights"])

    df_sorted = df.sort_values(["weights"], ascending=[False])
    df_sorted.to_csv('./inference/sorted_weights_{0}_{1}.csv'.format(name.replace(' ', '-'), model_type), mode='w')
    print(df_sorted)

if __name__ == "__main__":
    weight_sort("hide on bush", "enemy")
