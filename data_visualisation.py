import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def main():
    try:
        my_data = pd.read_csv(sys.argv[1])
        x = my_data.loc[:, 'di':'fi30']
        sns.pairplot(data=x, hue='di')
        plt.show()

    except:
        print('Please add new_data.csv and try again')


if __name__ == '__main__':
    main()
