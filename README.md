# AI benchmark diligence: how random splits overstate AI-for-chemistry accuracy

[![tests](https://github.com/f20220970-debug/ai-benchmark-diligence/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/ai-benchmark-diligence/actions/workflows/tests.yml)

A technical-diligence exercise for AI-for-science startups (drug discovery, materials, chemicals). A pitch deck says "our model hits R² 0.70". **The first question is: on which test set?** If molecules are split at random, near-duplicates of the training molecules end up in the test set and the score flatters the model. Splitting by **molecular scaffold** (test molecules share no Bemis–Murcko core with training) is closer to how the model will be used: on new chemistry.

Built by a chemistry + engineering student as part of applying for an AI venture-investing role: the same check applies to any AI company's accuracy claims.

## Results (mean of 5 splits each, public data)

| Dataset | Model | R², random split | R², scaffold split | RMSE change |
|---|---|---|---|---|
| **ESOL** solubility (1,128 molecules) | Random forest on Morgan fingerprints | **0.70** | **0.36** | **+43%** |
| | 4-descriptor linear model (Delaney-style) | 0.78 | 0.69 | +14% |
| **FreeSolv** hydration energy (642 molecules) | Random forest on Morgan fingerprints | **0.70** | **0.09** (±0.30) | **+102%** |
| | 4-descriptor linear model (Delaney-style) | 0.70 | 0.57 | +37% |

**Why it happens:** in a random split, 83% (ESOL) and 94% (FreeSolv) of test molecules share a scaffold with a training molecule; in the scaffold split, 0%. The median similarity of a test molecule to its nearest training molecule falls from 0.56 to 0.42 (ESOL) and from 0.53 to 0.34 (FreeSolv).

**What a diligence reader should take away:**
1. The same model's "R² 0.70" became **0.09 on new scaffolds** (FreeSolv). The headline number depended on the split, not the model.
2. A **4-variable linear model** in the spirit of Delaney's 2004 ESOL equation (logP, molecular weight, rotatable bonds, aromatic proportion) **beat the fingerprint random forest on new scaffolds** in both datasets. Chemistry-informed features generalise; memorised substructures don't. The baseline matters.
3. Scaffold-split scores vary a lot across splits (FreeSolv RF ±0.30), so a single split is not evidence.

## The checklist I'd use on any AI startup's accuracy claim
1. **How was the test set built?** Random, scaffold/cluster, time-based, or by customer? (For LLM agents: unseen wording; my [triage agent](https://github.com/f20220970-debug/llm-triage-agent) scores 96% on familiar wording and 72% on new wording.)
2. **How close is the test set to the training data?** Ask for nearest-neighbour similarity.
3. **What does a simple baseline score** on the same split?
4. **How much does the score move across splits?**
5. **Any prospective validation:** predictions made *before* the lab result came in?

## Run
```
pip install rdkit scikit-learn pandas pytest
python src/benchmark.py     # ~2 min: 2 datasets x 2 splits x 5 seeds -> docs/results.json
pytest -q                   # featurisation, scaffold split leaks no core into training, random split is a partition
```

<sub>Data: ESOL from Delaney, J. Chem. Inf. Comput. Sci. 2004 (via the DeepChem repository); FreeSolv v0.52 from Mobley & Guthrie, J. Comput. Aided Mol. Des. 2014 (MobleyLab/FreeSolv). Random forest uses 300 trees and is not tuned; a tuned model would score somewhat differently.</sub>
