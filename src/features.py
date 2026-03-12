from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def build_preprocessor(num_features, cat_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    # Quick sanity check for the preprocessor pipeline.
    # It should construct without errors when given a couple of example feature lists.
    preprocessor = build_preprocessor(num_features=["LotArea", "YearBuilt"], cat_features=["Neighborhood"])
    print(preprocessor)
