import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno

sns.set_theme()

def plot_count(df: pd.Series):
    ax = sns.countplot(x = df)
    ax.bar_label(ax.containers[0])
    plt.xlabel(df.name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {df.name}')
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

def pie_plot(df: pd.DataFrame, col: str):
    p = df[col].value_counts(normalize=True).mul(100).round(2)
    ax = plt.pie(p, labels = ['Not Exited', 'Exited'], autopct='%1.2f%%')

def dist_plot(df: pd.DataFrame, x: str, y: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(df, x = x, kde = True, hue = y, ax = ax1)
    sns.boxplot(df, y = x, x = y, orient='v', ax = ax2)