import pandas as pd
from data_generator1 import SampleDataGenerator

def build_dataset():
    # Instantiate the synthetic data generator
    data_gen = SampleDataGenerator()

    # Create raw synthetic inputs
    df = data_gen.syntetic_data_gen()

    # Add PHA yield using the dependent correlation logic
    df = data_gen.depended_correlation(df)
    print("Yield range:", df["yield"].min(), "to", df["yield"].max())



    # Print dataset summary
    print("\n✅ Synthetic dataset generated successfully!")
    print("Shape:", df.shape)
    print(df.head())

    # Save to file (optional)
    df.to_csv("synthetic_bioprocess_dataset1.csv", index=False)
    print("\n📁 Saved as synthetic_bioprocess_dataset1.csv")

    return df


if __name__ == "__main__":
    dataset = build_dataset()
