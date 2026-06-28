from src.data_io import load_processed_data
from src.network_analyzer import NetworkAnalyzer

def main():
    df = load_processed_data()

    network = NetworkAnalyzer(top_n_entities=50)
    network_data = network.generate_network(df)
    network.save_network(network_data)

    print("Network rebuilt")

if __name__ == "__main__":
    main()