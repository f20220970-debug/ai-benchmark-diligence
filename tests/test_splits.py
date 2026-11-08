import numpy as np, pandas as pd
from src.benchmark import featurise, scaffold_split, random_split

DF = pd.DataFrame({"smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "C1CCCCC1O", "C1CCCCC1N", "CCO", "CCCO",
                              "c1ccncc1", "c1ccncc1C", "not_a_smiles", "O=C(O)c1ccccc1", "CC(C)O"], "y": range(12)})

def test_invalid_smiles_dropped_and_features_align():
    df, X_fp, X_desc, fps, scaf = featurise(DF)
    assert len(df) == 11 == len(X_fp) == len(X_desc) == len(scaf)

def test_scaffold_split_shares_no_core_with_training():
    df, X_fp, X_desc, fps, scaf = featurise(DF)
    for seed in range(5):
        tr, te = scaffold_split(scaf, seed, frac=0.3)
        assert set(tr).isdisjoint(te) and len(tr) + len(te) == len(scaf)
        assert {scaf[i] for i in te}.isdisjoint({scaf[i] for i in tr})

def test_random_split_is_a_partition():
    tr, te = random_split(100, 0)
    assert len(te) == 20 and len(set(tr) | set(te)) == 100
