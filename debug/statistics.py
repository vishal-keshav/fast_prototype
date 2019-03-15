import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class distribution_plotter:
    def __init__(self):
        self.distribution_data = {}

    def add_distribution_data(self, data, name):
        self.distribution_data[name] = data

    def plot_distribution(self, title = 'plot', xlabel = 'xlable',
                                                             ylabel = 'ylabel'):
        for key, value in self.distribution_data.items():
            sns.distplot(value, hist = False, kde = True,
                        kde_kws = {'linewidth': 3}, label = key)
        plt.legend(prop={'size': 16}, title = title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

#---------Test---------

def main():
    dist1 = np.random.normal(-1, 1, 10000)
    dist2 = np.random.normal(1, 4, 10000)
    dist_plotter = distribution_plotter()
    dist_plotter.add_distribution_data(dist1, 'distribution1')
    dist_plotter.add_distribution_data(dist2, 'distribution2')
    dist_plotter.plot_distribution()

if __name__ == "__main__":
    main()
