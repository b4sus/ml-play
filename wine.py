import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    wines = pd.read_csv("data/wine/winequality-white.csv", sep=";")
    print(wines["quality"].value_counts())
    # wines.hist()
    quality = wines["quality"].to_numpy()
    plt.hist(quality, bins=50)
    plt.show()