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

train_df = pd.read_csv('path/datos_zenodo/train.csv')
train_df["Patch filename"]=train_df["Patch filename"].apply(append_ext)

val_df = pd.read_csv('path/datos_zenodo/val.csv')
val_df["Patch filename"]=val_df["Patch filename"].apply(append_ext)

test_df = pd.read_csv('path/datos_zenodo/test.csv')
test_df["Patch filename"]=test_df["Patch filename"].apply(append_ext)

grx_data = {"data_dir": "/path/GRX/Normalized/",
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df}
