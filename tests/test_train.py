import unittest
import os
import joblib
from app.train import model

class TestTrain(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(os.path.isfile('model/model.pkl'))

    def test_model_load(self):
        loaded_model = joblib.load('model/model.pkl')
        self.assertIsNotNone(loaded_model)

if __name__ == '__main__':
    unittest.main()