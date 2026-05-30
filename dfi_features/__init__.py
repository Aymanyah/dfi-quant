"""dfi_features: pure feature implementations.

Each feature is a function `compute(df, params) -> pd.Series` declared in its
own module. The runner in `agents/feature_construction/` discovers them via
the registry in `configs/features.yaml`.
"""
