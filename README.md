# NBA Moneyball

What is each NBA player actually worth, compared to what he gets paid?

Seasons 2010-11 onwards, data from stats.nba.com via nba_api.

## Plan

1. Pull play-by-play and split every game into stints (same 10 players on the floor)
2. RAPM: ridge regression of stint margin on who was playing
3. Box-prior RAPM: shrink towards a box score estimate instead of zero
4. Convert to wins
5. Aging curve, to project future wins
6. Injury discount
7. Join salaries, compute surplus value (projected value minus salary)
8. Sub-question: are three-point shooters overpaid?

## Setup

```
pip install -r requirements.txt
python src/check_api.py
```

stats.nba.com blocks most cloud IPs, so run the pulls from your own machine.
