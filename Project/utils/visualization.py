import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno

sns.set_theme()

def plot_count(df: pd.DataFrame, col: str):
    ax = sns.countplot(x = df[col])
    ax.bar_label(ax.containers[0])
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {col}')
    plt.show()

def plot_missing(df: pd.DataFrame):
    msno.matrix(df)

def crosstab_plot(df: pd.DataFrame, col: str, target: str):
    ct = pd.crosstab(df[col], df[target])
    ax = ct.plot.bar(stacked = True)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.show()
