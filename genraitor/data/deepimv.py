from pathlib import Path
import duckdb


def top_shapley(experiment_path: Path):
    data_path = [
        d
        for d in experiment_path.iterdir()
        if d.suffix == ".csv"
        and "AH1" in d.name
        and d.name.startswith("shap")
        and "pro" in d.name
    ]
    if len(data_path) == 0:
        raise FileExistsError(f"shapely file not found: {experiment_path}")
    data_path = data_path[0]

    data = duckdb.read_csv(str(data_path))
    duckdb.register("data", data)
    query = """
        SELECT
            'AH1' as pathogen
            ,feature as name
            ,shapley_value as shapley
            ,rank() over(order by shapley_value desc) as rank
        FROM data
        WHERE feature != 'Time'
        LIMIT 20 
    """
    return duckdb.sql(query).df()
