import pandas as pd
from src.tacoma.features import load_and_prepare_data

def test_feature_columns():
    df = pd.DataFrame({
        "date time": pd.date_range("2024-01-01", periods=48, freq="H"),
        "kWh": range(48)
    })
    path = "tmp_test.csv"
    df.to_csv(path, index=False)
    out = load_and_prepare_data(path, usage_col="kWh")
    for col in [
        "date","day_of_week","day_of_month","day_of_year",
        "hour_of_day","hour_of_week","hour_of_month","hour_of_year",
        "is_winter","is_weekend","season","day_type","season_daytype"
    ]:
        assert col in out.columns
