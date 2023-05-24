import pandas as pd
from scipy.stats import chi2_contingency
import itertools as it

class FeatureAssociation:
    def __init__(self, df: pd.DataFrame, alpha: float = 0.05):
        self.df = df
        self.alpha = alpha
        self.chi2_test()

    def chi2_test(self):
        self.data = {'Significatnt': [], 'Not Significant': []}
        col_prod = it.combinations(self.df.columns, 2)
        for col1, col2 in col_prod:
            if col1 != col2:
                ct = pd.crosstab(self.df[col1], self.df[col2])
                chi2, p, dof, expected = chi2_contingency(ct)
                if p < self.alpha:
                    self.data['Significatnt'].append((col1, col2, f'{p:.4f}'))
                else:
                    self.data['Not Significant'].append((col1, col2, p))

    def get_df(self, significant: bool = True):
        print('Significant Associations' if significant else 'Non Significant Associations')
        return pd.DataFrame(self.data['Significatnt' if significant else 'Not Significant'], columns=['Feature 1', 'Feature 2', 'p-value'])