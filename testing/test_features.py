"""
Code Style  : https://peps.python.org/pep-0008/ 
unittest    : https://docs.python.org/2/library/unittest.html#basic-example
"""
import unittest
import pandas as pd
from features import FeatureEngineer

# Rename Column
class TestRenameColumn(unittest.TestCase):
    """
    - Columns renamed correctly
    - Column names are lowercased
    - Original dataframe is not modified
    - Empty rename_map case
    """
    
    def test_rename_and_lower_columns(self):
        df = pd.DataFrame(columns=["Age", "User Name"])
        rename_map = {"Age": "age", "User Name": "username"}

        result = FeatureEngineer.rename_column(df=df, rename_map=rename_map)

        self.assertListEqual(
            list(result.columns),
            ["age", "username"]
        )

    def test_original_dataframe_not_modified(self):
        df = pd.DataFrame(columns=["Age"])
        rename_map = {"Age": "age"}

        FeatureEngineer.rename_column(df, rename_map)

        self.assertIn("Age", df.columns)

if __name__ == '__main__':
    unittest.main()

# Build Profile Text