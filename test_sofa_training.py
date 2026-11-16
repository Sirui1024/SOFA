import unittest

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from SOFA import fit_search_model


class SearchTrainingTests(unittest.TestCase):
    def test_empty_training_preserves_random_fallback(self):
        model = GradientBoostingRegressor(random_state=0)
        self.assertFalse(fit_search_model(model, [], []))
        self.assertFalse(hasattr(model, 'estimators_'))

    def test_empty_refit_preserves_fitted_model(self):
        model = GradientBoostingRegressor(random_state=0)
        designs = [[0.25, 0.75], [0.5, 0.5]]
        self.assertTrue(fit_search_model(model, designs, [0.0, 1.0]))
        before = model.predict(designs)
        self.assertFalse(fit_search_model(model, [], []))
        np.testing.assert_array_equal(model.predict(designs), before)


if __name__ == '__main__':
    unittest.main()
