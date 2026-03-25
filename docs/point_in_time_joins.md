# Point-in-Time Joins in V2

## Problem
If training rows use future information, the model gets unrealistically strong offline metrics. This is called temporal leakage.

## Solution
For each impression:
- join the latest user features with timestamp <= impression_time
- join the latest item features with timestamp <= impression_time

## Example
If a user saw an article at 10:00 AM, the joined features must reflect only what was known up to 10:00 AM.

## V2 implementation
- `build_impressions_v2.py` creates the impression backbone
- `build_historical_features_v2.py` builds timestamped user/item feature tables
- `point_in_time_join_v2.py` performs backward as-of joins
- `check_temporal_validity_v2.py` validates that no future features were joined