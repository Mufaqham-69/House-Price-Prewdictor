"""Utility script to build and print the preprocessor pipeline.

Run from the workspace root:

    python src/run_preprocessor.py
"""

from features import build_preprocessor


def main() -> None:
    preprocessor = build_preprocessor(
        num_features=["LotArea", "YearBuilt"],
        cat_features=["Neighborhood"],
    )
    print(preprocessor)


if __name__ == "__main__":
    main()
