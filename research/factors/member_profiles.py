"""Pure, contemporaneous factors for the selected exploratory member study.

Inputs are the union of top-20 long, short and volume lists, as wide tables.
An absent holding side is zero only for visible-position comparisons. Imputation
variants are explicit sensitivity analyses; none modifies the input records.
No returns, positions, execution logic, or market-file access belongs here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.factors.member import (
    daily_position_flow, disclosed_change_sum, net_gross_share, seat_turnover_groups, sticky_net_share,
)


FACTOR_META: dict[str, dict[str, str | int]] = {}


def _register(name: str, family: str, description: str, sign: int = 1) -> None:
    FACTOR_META[name] = {'family': family, 'sign': sign, 'description': description}


for _name, _description in (
    ('STICKY_NET_SHARE', 'Visible net/gross of seats below market turnover, including undisclosed volume.'),
    ('SPEC_NET_SHARE', 'Visible net/gross of seats at or above market turnover.'),
):
    _register(_name, 'baseline', _description)

for _name, _description in (
    ('RUN_LONG', 'Net/gross of currently held seats present for at least 20 consecutive ranked days.'),
    ('DIR_STABLE', 'Net/gross with absolute mean visible-net sign at least .8 over 20 consecutive holding days.'),
):
    _register(_name, 'profile', _description)

for _name, _description in (
    ('FLOW_CONTROL', 'Disclosed aggregate long change minus short change / market open interest.'),
    ('FLOW_CHANGE5', 'FLOW_CONTROL minus its value five sessions earlier.'),
    ('FLOW_MA5', 'Five-day mean of FLOW_CONTROL with five valid observations.'),
    ('NEG_SHORT_FLOW', 'Negative disclosed aggregate short change / market open interest.'),
):
    _register(_name, 'flow', _description)

for _pct in ('00', '50', '80', '100'):
    for _group in ('STICKY', 'SPEC'):
        _register(f'IMP{_pct}_{_group}', 'imputation',
                  f'Net/gross after missing-side fill at {_pct}% of its daily disclosed minimum; original {_group.lower()} membership fixed.')


def _impute(long: pd.DataFrame, short: pd.DataFrame,
            fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fraction == 0:
        # The literal legacy comparator is defined even if a whole side list is absent.
        return long.fillna(0.), short.fillna(0.)
    long_fill = long.min(axis=1) * fraction
    short_fill = short.min(axis=1) * fraction
    filled_long = long.mask(long.isna() & short.notna(), long_fill, axis=0)
    filled_short = short.mask(short.isna() & long.notna(), short_fill, axis=0)
    return filled_long, filled_short


def profile_factors(tables: dict[str, pd.DataFrame], market: pd.DataFrame) -> pd.DataFrame:
    """Return the selected exploratory factors on market's ordered dates without I/O.

    ``ranked`` is a date-indexed Series. Other tables have date rows and seat
    columns. Market requires finite, positive ``oi`` and ``volume``. Missing
    ranked sessions break run and direction histories. Empty groups and unavailable
    classifications remain NaN; neutral net positions remain zero. Imputation
    retains the original visible-position turnover groups for every fraction.
    """
    index = market.index
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('market dates must be unique and increasing')
    columns = tables['L'].columns
    for key in ('S', 'V', 'dL', 'dS'):
        columns = columns.union(tables[key].columns, sort=False)
    ranked = tables['ranked'].reindex(index).fillna(False).astype(bool)
    t = {key: tables[key].reindex(index=index, columns=columns).astype(float)
         for key in ('L', 'S', 'V', 'dL', 'dS')}
    t = {key: value.where(np.isfinite(value)).where(ranked, axis=0)
         for key, value in t.items()}
    t['ranked'] = ranked
    m = market[['oi', 'volume']].astype(float)
    m = m.where(np.isfinite(m))
    valid = ranked & m['oi'].gt(0) & m['volume'].gt(0)
    long, short = t['L'], t['S']
    net, gross, sticky, spec = seat_turnover_groups(t, m)
    held = gross.gt(0) & ranked.to_numpy()[:, None]
    presence = (long.notna() | short.notna()) & ranked.to_numpy()[:, None]
    out = {name: value for name, value in sticky_net_share(t, m).items()}

    # A missing ranked day and an absent seat both reset this day's run to zero.
    streak = np.zeros(presence.shape, dtype=np.int64)
    present_array = presence.to_numpy()
    for row in range(len(index)):
        previous = streak[row - 1] if row else 0
        streak[row] = np.where(present_array[row], previous + 1, 0)
    run = pd.DataFrame(streak, index=index, columns=columns)
    out['RUN_LONG'] = net_gross_share(net, gross, held & run.ge(20))
    direction = np.sign(net).where(presence).rolling(20, min_periods=20).mean().abs()
    out['DIR_STABLE'] = net_gross_share(net, gross, held & direction.ge(.8))

    for pct in (0, 50, 80, 100):
        filled_long, filled_short = _impute(long, short, pct / 100)
        filled_net, filled_gross = filled_long - filled_short, filled_long + filled_short
        for group, mask in (('STICKY', sticky), ('SPEC', spec)):
            out[f'IMP{pct:02d}_{group}'] = net_gross_share(filled_net, filled_gross, mask, require_complete=True)
    flow = daily_position_flow(t, m['oi']).where(valid)
    out['FLOW_CONTROL'] = flow
    out['FLOW_MA5'] = flow.rolling(5, min_periods=5).mean()
    out['FLOW_CHANGE5'] = flow.diff(5)
    oi = m['oi'].where(m['oi'].gt(0))
    out['NEG_SHORT_FLOW'] = -disclosed_change_sum(t, 'S') / oi
    return pd.DataFrame(out, index=index).reindex(columns=FACTOR_META).where(valid, axis=0)
