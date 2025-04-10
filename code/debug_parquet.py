import os
from save_and_load_parquet import SaveAndLoadParquet


OUTPUT_DIR = "D:/mimic/processed"

sl = SaveAndLoadParquet()
# train_df = sl.load_from_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
# test_df = sl.load_from_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))

train_df = sl.load_from_parquet(os.path.join(OUTPUT_DIR, "parsed_train.parquet"))
test_df = sl.load_from_parquet(os.path.join(OUTPUT_DIR, "parsed_test.parquet"))

print(train_df.columns.to_list())
print(test_df.columns.to_list())