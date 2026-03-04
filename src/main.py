import os
from preprocess import merge_magnifications, split_dataset
from feature_extraction import extract_features
from mbo import run_mbo
from classifier import train_and_evaluate


def main():

    raw_path = "data/raw"
    working_path = "data"
    split_root = "data/split"

    print("Merging magnifications...")
    benign, malignant = merge_magnifications(raw_path, working_path)

    print("Splitting dataset...")
    train_path, test_path = split_dataset(benign, malignant, split_root)

    print("Extracting features...")
    train_df = extract_features(train_path)
    test_df = extract_features(test_path)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = train_df.drop(columns=["filename", "label"]).values
    y_train = train_df["label"].values

    X_test = test_df.drop(columns=["filename", "label"]).values
    y_test = test_df["label"].values

    print("Running MBO...")
    selected_features = run_mbo(X_train, y_train)

    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    print("Training SVC...")
    train_and_evaluate(X_train_sel, y_train, X_test_sel, y_test, selected_features)


if __name__ == "__main__":
    main()