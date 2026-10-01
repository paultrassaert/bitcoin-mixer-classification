"""Provider-independent feature extraction from explicitly supplied transaction records.

Amounts are integer satoshis; scripts are strings such as P2TR/P2WPKH.
No private API, transaction data, assignment template or credentials are distributed.
"""
from collections import Counter
import math
import numpy as np

COMMON_DENOMINATIONS = {100000, 1000000, 10000000, 100000000}


def transaction_features(record: dict) -> dict:
    ins, outs = record['input_values_sats'], record['output_values_sats']
    if not ins or not outs:
        raise ValueError('Non-coinbase transactions require inputs and outputs')
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in ins+outs):
        raise ValueError('Amounts must be non-negative integer satoshis')
    if record['vsize'] <= 0:
        raise ValueError('vsize must be positive')
    input_scripts, output_scripts = record['input_scripts'], record['output_scripts']
    if len(input_scripts) != len(ins) or len(output_scripts) != len(outs):
        raise ValueError('Script/value lengths differ')
    ni, no, vsize = len(ins), len(outs), record['vsize']
    total_in, total_out = sum(ins), sum(outs)
    if total_out > total_in:
        raise ValueError('Negative transaction fee: incomplete or inconsistent input data')
    counts, scripts = Counter(outs), Counter(output_scripts)
    entropy = -sum((count/no)*math.log2(count/no) for count in counts.values())
    common = sum(v in COMMON_DENOMINATIONS for v in outs)
    first_non_denom = next((i for i,v in enumerate(outs) if v not in COMMON_DENOMINATIONS), None)
    fee = total_in-total_out
    return {
        'txid': record['txid'], 'n_inputs': ni, 'n_outputs': no,
        'input_output_ratio': ni/no, 'vsize': vsize, 'vsize_per_output': vsize/no,
        'total_input_btc': total_in/1e8, 'fee_btc': fee/1e8, 'fee_rate': fee/vsize,
        'entropy': entropy, 'output_cv': float(np.std(outs)/np.mean(outs)) if total_out else 0.,
        'unique_out_vals': len(counts), 'unique_output_ratio': len(counts)/no,
        'max_equal_output_count': max(counts.values()),
        'outputs_at_common_denom': common, 'common_denom_ratio': common/no,
        'p2tr_ratio': scripts.get('P2TR',0)/no, 'p2wpkh_ratio': scripts.get('P2WPKH',0)/no,
        'input_homogeneity': max(Counter(input_scripts).values())/ni,
        'n_locktime': record.get('locktime',0), 'is_locktime_set': int(record.get('locktime',0)>0),
        # This is a denomination-pattern proxy, not identification of a change output.
        'first_nondenom_relative_position': first_non_denom/no if first_non_denom is not None else -1.,
        'has_nondenom_output': int(first_non_denom is not None)
    }
