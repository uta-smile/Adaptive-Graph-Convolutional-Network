
import numpy as np

from AGCN.utils.splitter import Splitter


class RandomSplitter(Splitter):
    """
    Child Class for doing random data splits.
    """
    def split(self,
            dataset,
            seed=None,
            frac_train=.5,
            frac_valid=.2,
            frac_test=.3,
            log_every_n=None):
        """
        Splits internal compounds randomly into train/validation/test.
        """
        np.testing.assert_almost_equal(float(frac_train + frac_valid + frac_test), 1.)

        if not seed is None:
            np.random.seed(seed)
        num_datapoints = len(dataset)
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)
        shuffled = np.random.permutation(range(num_datapoints))
        return shuffled[:train_cutoff], shuffled[train_cutoff:valid_cutoff], shuffled[valid_cutoff:]