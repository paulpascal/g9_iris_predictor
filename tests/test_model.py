import unittest
import joblib
import pandas as pd
from sklearn.datasets import load_iris


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("model/model.pkl")
        iris = load_iris()
        self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
        # Change column names
        self.X.rename(
            columns={
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
            },
            inplace=True,
        )
        self.y = pd.Series(iris.target)

    def test_model_predict(self):
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))


if __name__ == "__main__":
    unittest.main()
