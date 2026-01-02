from datasets import load_dataset

def main():
    ds = load_dataset("laolao77/MMDU", split="train")
    print("Columns:", ds.column_names)
    print("Sample[0]:", ds[0])

    small = ds.select(range(min(50, len(ds))))
    print("Small subset size:", len(small))

if __name__ == "__main__":
    main()
