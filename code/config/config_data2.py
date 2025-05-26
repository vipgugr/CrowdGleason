import pandas as pd

######  Data VLC
sicap_data = {"data_dir":"/path/SICAPv2/Norm_T0/",
	      "train_df": pd.read_excel("/path/SICAPv2/partition/Validation/Val1/Train.xlsx"),
	      "val_df": pd.read_excel('/path/SICAPv2/partition/Validation/Val1/Test.xlsx'),
	      "test_df": pd.read_excel('/path/SICAPv2/partition/Test/Test.xlsx')}

####### Data GRX
# In the normalized images the extension of GRX is .png
def append_ext(fn):
    return fn+".png"

train_df_ = pd.read_csv('/path/crowdgleason/train.csv')
train_df_["Patch filename"]=train_df_["Patch filename"].apply(append_ext)
crowd_train_df = train_df_.copy()
one_hot_mv_train = pd.get_dummies(train_df_["label"])
train_df_mv = pd.concat([train_df_["Patch filename"], one_hot_mv_train], axis=1)

test_df_ = pd.read_csv('/path/crowdgleason/test.csv')
test_df_["Patch filename"]=test_df_["Patch filename"].apply(append_ext)

condition = (test_df_['ground truth'] == test_df_['label'])

# Use boolean indexing to subset the DataFrame
consensus_df = test_df_[condition]
print(consensus_df)


one_hot = pd.get_dummies(consensus_df["ground truth"])
test_df = pd.concat([consensus_df["Patch filename"], one_hot], axis=1)

# VALIDATION

val_df_ = pd.read_csv('/path/crowdgleason/val.csv')
val_df_["Patch filename"]=val_df_["Patch filename"].apply(append_ext)
one_hot = pd.get_dummies(val_df_["label"])
val_df_mv = pd.concat([val_df_["Patch filename"], one_hot], axis=1)



grx_data = {"data_dir_train":  "/path/crowdgleason/train/",
            "data_dir_test": "/path/crowdgleason/test/",
            "train_df_mv": train_df_mv,
			"val_df_mv": val_df_mv,
            "test_df": test_df}
