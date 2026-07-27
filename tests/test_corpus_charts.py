import pandas as pd

from ui.corpus_charts import build_source_hit_bar, build_source_trends_line, build_trends_line


def test_build_charts_empty_and_ok():
    assert build_source_hit_bar(pd.Series(dtype=int)) is None
    assert build_trends_line(pd.DataFrame(columns=["bucket", "term", "count"])) is None
    fig = build_trends_line(
        pd.DataFrame(
            {
                "bucket": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "term": ["футбол", "футбол"],
                "count": [1, 2],
            }
        )
    )
    assert fig is not None
    fig2 = build_source_trends_line(
        pd.DataFrame(
            {
                "bucket": [pd.Timestamp("2024-01-01")],
                "source": ["A"],
                "count": [3],
            }
        )
    )
    assert fig2 is not None
    bar = build_source_hit_bar(pd.Series({"A": 2, "B": 1}))
    assert bar is not None
