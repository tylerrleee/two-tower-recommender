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
        FE = FeatureEngineer(
            numeric_fields=["age", "going_out"]
        )

        df = pd.DataFrame({
            "age"       : ["20", "bad", None], 
            "going_out" : ["10.0", "5.0", None],
            "name"      : ["Alice", None, "Bob"]
        })

        result = FE._clean_table(df)

        self.assertEqual(result["age"].tolist(), [20.0, 0.0, 0.0])
        self.assertEqual(result["going_out"].tolist(), [10.0, 5.0, 0.0])
        self.assertEqual(result["name"].tolist(), ["Alice", "", "Bob"])

class TestFit(unittest.TestCase):
    """
    - ColumnTransformer is created
    - fitted flag is setr
    - Correct number of output features
    - Error when there are no valid features
    """
    def test_fit_sets_fitted(self):
        """
        Once perform fitting, the variable instance are updated accordingly
        - Fitted is True 
        - ColumnTransformer is updated
        """
        df = pd.DataFrame({
            "gender"    : ["M", "F"],
            "age"       : [20, 30]
        })

        FE = FeatureEngineer(
            profile_text        = [],
            categorical_fields  = ["gender"],
            numeric_fields      = ["age"]
        )

        FE.fit(df=df,
               rename_map=None)
        
        self.assertTrue(FE.fitted)
        self.assertIsNotNone(FE.column_transformer)
    
    def test_fit_raises_if_no_features(self):
        """
        Given missing features, does fitting it raise a ValueError?
        """
        df = pd.DataFrame({"a": [1, 2]})

        fe = FeatureEngineer(
            profile_text        = [],
            categorical_fields  = ["missing"],
            numeric_fields      = ["also_missing"]
        )

        with self.assertRaises(ValueError):
            fe.fit(df, rename_map=None)

class TestTransform(unittest.TestCase):
    """
    - Raises if not fitted
    - Output keys exist
    - Output arrays shapes are consistent
    - Meta Features are numeric only
    """
    def test_transform_before_fit_raises(self):
        fe = FeatureEngineer(
            profile_text        =["bio"],
            categorical_fields  =["gender"],
            numeric_fields      =["age"]
        )

        df = pd.DataFrame({
            "bio"   : ["Hello"],
            "gender": ["M"],
            "age"   : [20]
        })

        with self.assertRaises(RuntimeError):
            fe.transform(df)
    
    def test_transform_output_structure(self):
        """
        - number of profile text and metadata should match the number of apps
        - Once fitted, we can transform the df
        - result should have 4 key outputs
        """
        df = pd.DataFrame({
            "bio"   : ["Hello", "Hi"],
            "gender": ["M", "F"],
            "age"   : [20, 30]
        })

        fe = FeatureEngineer(
            profile_text        =["bio"],
            categorical_fields  =["gender"],
            numeric_fields      =["age"]
        )

        fe.fit(df, rename_map=None)
        result = fe.transform(df)

        self.assertIn("profile_text", result)
        self.assertIn("meta_features", result)
        self.assertIn("index", result)
        self.assertIn("raw_df", result)

        n_applicants = len(df['bio'])
        self.assertEqual(len(result["profile_text"]), n_applicants) # 2 applicants
        self.assertEqual(result["meta_features"].shape[0], n_applicants) # 2 applicants

class TestDiversityFeatures(unittest.TestCase):

    def test_compute_diversity_features(self):
        df = pd.DataFrame({
            "extroversion": [0.5, 1.0],
            "study_frequency": [3, 5],
            "gym_frequency": [3, 1]
        })

        FE = FeatureEngineer()

        result = FE.compute_diversity_features(df)

        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result[0, 0], 1.0)
        self.assertEqual(result[1, 1], 4)

if __name__ == '__main__':
    unittest.main()

# Build Profile Text