from typing import Dict, List

try:
    import pandas as pd
except Exception:
    pd = None


def fit_encoder(df, categorical_cols: List[str]) -> Dict:
    """Fit simple encodings for categorical columns. Returns mapping.

    If pandas is available, store unique values per column.
    """
    mapping = {}
    if pd is not None and hasattr(df, 'loc'):
        for c in categorical_cols:
            if c in df.columns:
                mapping[c] = [None if pd.isna(v) else v for v in pd.unique(df[c].astype(object))]
            else:
                mapping[c] = []
    else:
        # df expected to be list of dicts
        for c in categorical_cols:
            vals = []
            for r in df:
                v = r.get(c)
                if v not in vals:
                    vals.append(v)
            mapping[c] = vals
    return mapping


def transform(df, mapping: Dict, categorical_cols: List[str]):
    """Transform dataframe or list-of-dicts using provided mapping.

    If pandas available, use one-hot encoding for mapped categories (columns prefixed).
    Otherwise, apply label indices.
    """
    if pd is not None and hasattr(df, 'loc'):
        out = df.copy()
        for c in categorical_cols:
            if c not in out.columns:
                continue
            vals = mapping.get(c, [])
            # one-hot
            for v in vals:
                colname = f'{c}__{v}'
                out[colname] = (out[c] == v).astype(int)
            out.drop(columns=[c], inplace=True)
        return out
    else:
        out = []
        for r in df:
            nr = r.copy()
            for c in categorical_cols:
                vals = mapping.get(c, [])
                v = nr.pop(c, None)
                if v in vals:
                    nr[f'{c}_idx'] = vals.index(v)
                else:
                    nr[f'{c}_idx'] = -1
            out.append(nr)
        return out
