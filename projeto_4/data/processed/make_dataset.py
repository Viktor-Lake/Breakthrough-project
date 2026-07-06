import polars as pl


def convert_csv_to_parquet(input_path: str, output_path: str):

    # importação dos dados 
    
    df = (
        pl.scan_csv(input_path)
    )

    # coversão de csv para parquet
    df.sink_parquet(output_path)
    

if __name__ == "__main__":
    convert_csv_to_parquet("data/raw/application_train.csv", "data/processed/application_train.parquet")
    