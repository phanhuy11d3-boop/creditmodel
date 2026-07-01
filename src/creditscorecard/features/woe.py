"""Weight-of-Evidence (WoE) / Information Value (IV) transformer.

Orientation is fixed by domain convention (spec constraint 4)::

    WoE = ln(%Good / %Bad)

so a **positive** WoE means a better applicant. With the target coded
``1 == Bad``, "Good" is ``y == 0``. WoE and IV are computed on the **train**
sample only and then frozen; ``transform`` is a pure lookup that never refits.

Zero cells are stabilised by adding 0.5 to any bin whose Good or Bad count is
zero (standard practice) so WoE stays finite. Bin codes never seen in training
map to WoE 0 (neutral) and are logged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditscorecard.features.binning import BinningModel
from creditscorecard.logging import get_logger

logger = get_logger(__name__)


def compute_woe_iv(codes: pd.Series, y: pd.Series) -> tuple[dict[int, float], float, pd.DataFrame]:
    """Compute the WoE map, IV, and a detail table for one binned characteristic.

    ``y`` uses the modeling convention ``1 == Bad``. Returns ``(woe_map, iv, table)``.
    """
    df = pd.DataFrame({"code": np.asarray(codes), "bad": np.asarray(y).astype(int)})
    df["good"] = 1 - df["bad"]
    grp = df.groupby("code", sort=True).agg(good=("good", "sum"), bad=("bad", "sum"))

    total_good = float(grp["good"].sum())
    total_bad = float(grp["bad"].sum())
    if total_good == 0 or total_bad == 0:
        raise ValueError("WoE requires both Good and Bad observations in the sample")

    good = grp["good"].astype(float).copy()
    bad = grp["bad"].astype(float).copy()
    zero_cell = (good == 0) | (bad == 0)
    good[zero_cell] += 0.5
    bad[zero_cell] += 0.5

    good_arr = good.to_numpy(dtype=float)
    bad_arr = bad.to_numpy(dtype=float)
    dist_good = good_arr / good_arr.sum()
    dist_bad = bad_arr / bad_arr.sum()
    woe = np.log(dist_good / dist_bad)
    iv_contrib = (dist_good - dist_bad) * woe
    iv = float(iv_contrib.sum())

    codes_arr = np.asarray(grp.index, dtype=int)
    table = pd.DataFrame(
        {
            "code": codes_arr,
            "count": (grp["good"] + grp["bad"]).to_numpy(),
            "good": grp["good"].to_numpy(),
            "bad": grp["bad"].to_numpy(),
            "bad_rate": (grp["bad"] / (grp["good"] + grp["bad"])).to_numpy(),
            "woe": woe,
            "iv": iv_contrib,
        }
    ).reset_index(drop=True)

    woe_map = {int(c): float(w) for c, w in zip(codes_arr.tolist(), woe.tolist(), strict=True)}
    return woe_map, iv, table


class WoETransformer:
    """Fit WoE/IV on train bin codes; transform is a frozen lookup."""

    def __init__(self, binning: BinningModel) -> None:
        self.binning = binning
        self.woe_maps: dict[str, dict[int, float]] = {}
        self.iv: dict[str, float] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.features: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoETransformer:
        codes = self.binning.transform(X)
        for col in codes.columns:
            woe_map, iv, table = compute_woe_iv(codes[col], y)
            self.woe_maps[col] = woe_map
            self.iv[col] = iv
            self.tables[col] = table
        self.features = list(codes.columns)
        self._fitted = True
        logger.info("Fitted WoE/IV on %d characteristics (train only).", len(self.features))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map bin codes to frozen WoE values (transform only, no refit)."""
        if not self._fitted:
            raise RuntimeError("WoETransformer.transform called before fit")
        codes = self.binning.transform(X)
        out = {}
        for col in self.features:
            wmap = self.woe_maps[col]
            mapped = codes[col].map(wmap)
            if mapped.isna().any():
                unseen = sorted(set(codes[col][mapped.isna()].unique().tolist()))
                logger.warning(
                    "Feature '%s': bin codes %s unseen in train; assigning WoE=0.",
                    col,
                    unseen,
                )
                mapped = mapped.fillna(0.0)
            out[f"woe_{col}"] = mapped.astype(float).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def iv_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"feature": list(self.iv.keys()), "iv": list(self.iv.values())})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def woe_maps_serialisable(self) -> dict[str, dict[str, float]]:
        return {f: {str(k): v for k, v in m.items()} for f, m in self.woe_maps.items()}
