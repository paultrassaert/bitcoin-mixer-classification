import unittest
from copy import deepcopy
import pandas as pd
from features import transaction_features
from train import check_data,EXPECTED_CLASSES


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.record={'txid':'synthetic','input_values_sats':[401000],
                     'output_values_sats':[100000]*4,'input_scripts':['P2WPKH'],
                     'output_scripts':['P2WPKH']*4,'vsize':200,'locktime':0}

    def test_equal_outputs_and_fee(self):
        f=transaction_features(self.record)
        self.assertEqual(f['entropy'],0)
        self.assertEqual(f['max_equal_output_count'],4)
        self.assertEqual(f['fee_rate'],5)
        self.assertEqual(f['first_nondenom_relative_position'],-1)

    def test_missing_sentinel_is_independent_of_output_count(self):
        r=deepcopy(self.record);r['output_values_sats']=[100000]*2;r['output_scripts']=['P2WPKH']*2
        self.assertEqual(transaction_features(r)['first_nondenom_relative_position'],-1)

    def test_invalid_amounts_rejected(self):
        r=deepcopy(self.record);r['input_values_sats']=[]
        with self.assertRaises(ValueError):transaction_features(r)
        r=deepcopy(self.record);r['output_values_sats'][0]=100000.5
        with self.assertRaises(ValueError):transaction_features(r)

    def test_cross_split_overlap_rejected(self):
        frame=pd.DataFrame({'txid':[f'id{i}' for i in range(5)],'label':sorted(EXPECTED_CLASSES),'entropy':[0.]*5})
        with self.assertRaisesRegex(ValueError,'overlap'):check_data(frame,frame.copy())


if __name__=='__main__':unittest.main()
