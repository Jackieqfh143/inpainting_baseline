from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns




class Ploter():
    def __init__(self,csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file,index_col= False)


    def draw_table(self,save_name):
        table_value = self.df.values
        col_names = self.df.columns
        plt.figure(figsize=(20, 8))
        tab = plt.table(cellText=table_value,
                        colLabels=col_names,
                        loc='center',
                        cellLoc='center',
                        rowLoc='center')
        tab.scale(1, 2)
        plt.axis('off')
        plt.savefig(save_name)

    def draw_scatter(self,x_name,y_name,circle_size_col,circel_name,save_name,title):
        plt.figure(figsize=(8, 8))
        x = self.df.loc[:,[x_name]].values
        y = self.df.loc[:,[y_name]].values
        s = 15 * self.df.loc[:,[circle_size_col]].values
        name_list = self.df.loc[:,[circel_name]].values

        rgb_values = sns.color_palette("pastel", s.size)

        plt.scatter(x, y, s=s,c=rgb_values)
        plt.title(title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)

        plt.ylim(bottom=int(np.min(y)-np.min(y)/6),top=int(np.max(y)+np.min(y)/6))
        plt.xlim(left=0,right=np.max(x)+np.min(x))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        for i, name in enumerate(name_list):
            plt.annotate(name.item(), (x[i], y[i]))


        plt.savefig(save_name)

    def save(self):
        self.df.sort_values('loss_mean', ascending=False, inplace=True)
        self.df.to_csv(self.csv_file, index=False)

