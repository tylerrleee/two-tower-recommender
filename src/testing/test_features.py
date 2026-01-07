"""
Code Style  : https://peps.python.org/pep-0008/ 
unittest    : https://docs.python.org/2/library/unittest.html#basic-example
"""
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from src.features.features import FeatureEngineer

# Rename Column
class TestRenameColumn(unittest.TestCase):
    """
    - Columns renamed correctly
    - Column names are lowercased
    - Original dataframe is not modified
    - Empty rename_map case
    """
    
    def test_rename_and_lower_columns(self):
        """
        Texts should be:
        - Renamed according to rename_map
        - Column names are lowered
        """
        df = pd.DataFrame(columns=["Age", "User Name"])
        rename_map = {"Age": "age", "User Name": "username"}

        result = FeatureEngineer.rename_column(df=df, rename_map=rename_map)

        self.assertListEqual(
            list(result.columns),
            ["age", "username"]
        )

    def test_None_and_NaN_values(self):
        """
        None values should be ignored
        """
        df = pd.DataFrame(columns=["Age", "User Name", None])
        rename_map = {"Age": "age", "User Name": "username"}

        result = FeatureEngineer.rename_column(df=df, rename_map=rename_map)

        self.assertListEqual(
            list(result.columns),
            ["age", "username", None]
        )

    def test_original_dataframe_not_modified(self):
        df = pd.DataFrame(columns=["Age"])
        rename_map = {"Age": "age"}

        FeatureEngineer.rename_column(df, rename_map)

        self.assertIn("Age", df.columns)
    
class TestProfileText(unittest.TestCase):
    def test_build_profile_test_core_function(self):
        """
        Text inputs are combined with period delimiter
        """
        df = pd.DataFrame({
            "bio": ["Hello", "Hi"],
            "hobby": ["Chess", "Music"]
        })

        result = FeatureEngineer.build_profile_text(
            df,
            text_fields=["bio", "hobby"]
        )

        self.assertEqual(result.iloc[0], "Hello. Chess")
        self.assertEqual(result.iloc[1], "Hi. Music")

    def test_build_profile_text_with_nan_and_none(self):
        """
        None and NaN values are ignored as they do not add value for now
        """
        df = pd.DataFrame({
            "bio": ["Hello", None],
            "hobby": [float("nan"), "Music"]
        })

        result = FeatureEngineer.build_profile_text(
            df,
            text_fields=["bio", "hobby"]
        )

        self.assertEqual(result.iloc[0], "Hello")
        self.assertEqual(result.iloc[1], "Music")

    def test_build_profile_text_missing_column(self):
        """
        Missing columns should have no effect
        - candidate_text = row.get(field, "") should lead to empty string for missing field
        """
        df = pd.DataFrame({"bio": ["Hello"]})

        result = FeatureEngineer.build_profile_text(
            df,
            text_fields=["bio", "missing"]
        )

        self.assertEqual(result.iloc[0], "Hello")

class TestCleanTable(unittest.TestCase):

    def test_clean_table_numeric_casting(self):
        """
        Numeric fields handles non-numeric cases
        - Non-numric are empty and therefore zero'd
        - other empty fields are casted into an empty string
        - Floats are treated the same
        ^ all numerics should be passed as a string, so we dont have to check floating numbers
        """
        fe = FeatureEngineer(
            numeric_fields=["age", "going_out"]
        )

        df = pd.DataFrame({
            "age": ["20", "bad", None], 
            "going_out": ["10.0", "5.0", None],
            "name": ["Alice", None, "Bob"]
        })

        result = fe._clean_table(df)

        self.assertEqual(result["age"].tolist(), [20.0, 0.0, 0.0])
        self.assertEqual(result["going_out"].tolist(), [10.0, 5.0, 0.0])
        self.assertEqual(result["name"].tolist(), ["Alice", "", "Bob"])

if __name__ == '__main__':
    unittest.main()

# Build Profile Text