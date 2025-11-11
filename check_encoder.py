import pandas as pd
import chardet
import csv


with open('capture_preprocessed.csv', 'rb') as f:
    rawdata = f.read(10000)
result = chardet.detect(rawdata)
encoding = result['encoding']
print("Detected encoding:", encoding)

with open('capture_preprocessed.csv', 'r', encoding=encoding) as f:
    sample = f.read(1024)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[',', '\t', ';', '|'])
        print("Detected delimiter:", dialect.delimiter)
    except csv.Error:
        print("Could not detect delimiter, trying common delimiters")
        # Test common delimiters manually
        for sep in [',', '\t', ';', '|']:
            print(f"Trying separator '{sep}':")
            try:
                example_df = pd.read_csv('capture_preprocessed.csv', sep=sep, encoding=encoding, nrows=5)
                print(example_df.head())
                break
            except Exception as e:
                print("Failed with error:", e)
