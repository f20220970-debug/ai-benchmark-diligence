"""How random splits overstate AI-for-chemistry accuracy, and what a diligence check looks like.

Two public datasets: ESOL (aqueous solubility, 1,128 molecules; Delaney 2004) and FreeSolv (hydration free energy,
642 molecules; Mobley & Guthrie 2014). For each, the same models are scored under two splits:
  - RANDOM split: what a pitch deck usually reports; near-duplicate molecules land in both train and test
  - SCAFFOLD split: test molecules share no Bemis-Murcko core with training molecules (closer to "new chemistry")
Models: random forest on Morgan fingerprints (the 'ML' a startup would show), and a 4-descriptor linear model in the
spirit of Delaney's 2004 ESOL equation (logP, molecular weight, rotatable bonds, aromatic proportion), refit on each
training split. Also measured: how similar each test molecule is to its nearest training molecule (Tanimoto), which
is the mechanism behind the gap.
"""
import json, warnings
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def load():
    e = pd.read_csv(ROOT / "data" / "esol_delaney.csv")
    esol = pd.DataFrame({"smiles": e["smiles"].str.strip(), "y": e["measured log solubility in mols per litre"]})
    rows = [l.split(";") for l in open(ROOT / "data" / "freesolv_database.txt") if not l.startswith("#") and l.strip()]
    fs = pd.DataFrame({"smiles": [r[1].strip() for r in rows], "y": [float(r[3]) for r in rows]})
    return {"ESOL (solubility, log mol/L)": esol, "FreeSolv (hydration free energy, kcal/mol)": fs}


def featurise(df):
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    keep = [m is not None for m in mols]
    df, mols = df[keep].reset_index(drop=True), [m for m in mols if m is not None]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    X_fp = np.array([list(f) for f in fps], dtype=np.uint8)
    def arom(m):
        return sum(a.GetIsAromatic() for a in m.GetAtoms()) / max(m.GetNumHeavyAtoms(), 1)
    X_desc = np.array([[Crippen.MolLogP(m), Descriptors.MolWt(m), rdMolDescriptors.CalcNumRotatableBonds(m), arom(m)] for m in mols])
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in mols]
    return df, X_fp, X_desc, fps, scaf


def random_split(n, seed, frac=0.2):
    idx = np.random.default_rng(seed).permutation(n); k = int(n * frac)
    return idx[k:], idx[:k]


def scaffold_split(scaf, seed, frac=0.2):
    """Whole scaffold groups go to one side. Big groups always train (as in MoleculeNet); the seed shuffles the
    order of small groups so results are averaged over several scaffold splits, not one lucky one."""
    groups = pd.Series(range(len(scaf))).groupby(pd.Series(scaf)).apply(list)
    rng = np.random.default_rng(seed)
    big = [g for g in groups if len(g) > len(scaf) * frac / 2]
    small = [g for g in groups if len(g) <= len(scaf) * frac / 2]
    rng.shuffle(small)
    test, cap = [], int(len(scaf) * frac)
    for g in small:
        if len(test) + len(g) <= cap:
            test += g
    test = set(test)
    train = [i for i in range(len(scaf)) if i not in test]
    return np.array(train), np.array(sorted(test))


def nn_similarity(fps, tr, te):
    tr_fps = [fps[i] for i in tr]
    return float(np.median([max(DataStructs.BulkTanimotoSimilarity(fps[i], tr_fps)) for i in te]))


def evaluate(df, X_fp, X_desc, fps, scaf, seeds=(0, 1, 2, 3, 4)):
    y = df.y.values; out = {}
    for split in ("random", "scaffold"):
        res = {k: [] for k in ("rf_rmse", "rf_r2", "lin_rmse", "lin_r2", "mean_rmse", "nn_sim", "shared_scaffold_pct")}
        for s in seeds:
            tr, te = random_split(len(y), s) if split == "random" else scaffold_split(scaf, s)
            rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=s).fit(X_fp[tr], y[tr])
            lin = LinearRegression().fit(X_desc[tr], y[tr])
            p_rf, p_lin = rf.predict(X_fp[te]), lin.predict(X_desc[te])
            res["rf_rmse"].append(mean_squared_error(y[te], p_rf) ** .5); res["rf_r2"].append(r2_score(y[te], p_rf))
            res["lin_rmse"].append(mean_squared_error(y[te], p_lin) ** .5); res["lin_r2"].append(r2_score(y[te], p_lin))
            res["mean_rmse"].append(mean_squared_error(y[te], np.full(len(te), y[tr].mean())) ** .5)
            res["nn_sim"].append(nn_similarity(fps, tr, te))
            tr_scaf = {scaf[i] for i in tr}
            res["shared_scaffold_pct"].append(100 * np.mean([scaf[i] in tr_scaf for i in te]))
        out[split] = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
        out[split]["rf_r2_sd"] = round(float(np.std(res["rf_r2"])), 3)
    out["rf_rmse_increase_pct"] = round(100 * (out["scaffold"]["rf_rmse"] / out["random"]["rf_rmse"] - 1), 1)
    out["n"] = int(len(y))
    return out


def main():
    results = {}
    for name, df in load().items():
        df, X_fp, X_desc, fps, scaf = featurise(df)
        results[name] = evaluate(df, X_fp, X_desc, fps, scaf)
        print(name, json.dumps(results[name], indent=1))
    json.dump(results, open(ROOT / "docs" / "results.json", "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
