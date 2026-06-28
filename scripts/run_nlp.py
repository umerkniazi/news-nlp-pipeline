from src.data_io import load_processed_data, save_processed_data
from src.nlp_processor import NLPProcessor

def main():
    df = load_processed_data()

    nlp = NLPProcessor()
    df = nlp.process_dataframe(df)

    save_processed_data(df)

    print("Done")

if __name__ == "__main__":
    main()