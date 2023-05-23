import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()

def plot_count(df: pd.DataFrame, col: str):
    ax = sns.countplot(x = df[col])
    ax.bar_label(ax.containers[0])
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {col}')
    plt.show()