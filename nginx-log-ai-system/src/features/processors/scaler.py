from typing import Dict, List

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None


def fit_scaler(df, numeric_cols: List[str]) -> Dict:
    params = {}
    if pd is not None and hasattr(df, 'loc'):
        for c in numeric_cols:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors='coerce')
                mean = float(col.mean()) if not col.dropna().empty else 0.0
                std = float(col.std(ddof=0)) if not col.dropna().empty else 1.0
                params[c] = {'mean': mean, 'std': std}
            else:
                params[c] = {'mean': 0.0, 'std': 1.0}
    else:
        # df is list of dicts
        for c in numeric_cols:
            vals = []
            for r in df:
                v = r.get(c)
                try:
                    v = float(v)
                    vals.append(v)
                except Exception:
                    continue
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = var ** 0.5
                params[c] = {'mean': mean, 'std': std}
            else:
                params[c] = {'mean': 0.0, 'std': 1.0}
    return params


def transform(df, params: Dict, numeric_cols: List[str]):
    if pd is not None and hasattr(df, 'loc'):
        out = df.copy()
        for c in numeric_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce')
                mean = params.get(c, {}).get('mean', 0.0)
                std = params.get(c, {}).get('std', 1.0) or 1.0
                out[c] = (out[c] - mean) / std
        return out
    else:
        out = []
        for r in df:
            nr = r.copy()
            for c in numeric_cols:
                v = nr.get(c)
                try:
                    v = float(v)
                    mean = params.get(c, {}).get('mean', 0.0)
                    std = params.get(c, {}).get('std', 1.0) or 1.0
                    nr[c] = (v - mean) / std
                except Exception:
                    nr[c] = None
            out.append(nr)
        return out
