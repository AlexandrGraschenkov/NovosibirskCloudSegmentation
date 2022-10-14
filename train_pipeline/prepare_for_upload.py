import pandas as pd


def prepare(orig_path, pred_path: str):
    out_path = pred_path.replace("_preds_", "_upload_")
    orig = pd.read_csv(orig_path)
    pred = pd.read_csv(pred_path)
    pred.set_axis(['Class'], axis=1, inplace=True)

    df_res = orig.join(pred)
    df_res = df_res[["id", "Class"]]
    df_res.to_csv(out_path, sep=',', index=False)


prepare("/home/anvar/Novosib/dataset/test_dataset_test.csv_result",
        "/home/anvar/Novosib/preds/test_preds_hd200_elu_XY_no_tr_9.csv")
