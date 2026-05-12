import logging
from src.data_io import load_raw_data, save_processed_data
from src.nlp_processor import NLPProcessor
from src.topic_modeler import TopicModeler
from src.network_analyzer import NetworkAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("Loading raw data...")
    df = load_raw_data()
    
    logger.info("Running NLP Processor (Sentiment & NER)...")
    nlp_processor = NLPProcessor()
    df = nlp_processor.process_dataframe(df, text_column='headline')
    
    logger.info("Running Topic Modeler (LDA)...")
    topic_modeler = TopicModeler()
    topic_modeler.fit(df['headline'].tolist())
    df = topic_modeler.assign_topics(df, text_column='headline')
    
    logger.info("Running Network Analyzer (Entity Relationships)...")
    network_analyzer = NetworkAnalyzer(top_n_entities=50)
    network_data = network_analyzer.generate_network(df)
    
    logger.info("Saving artifacts...")
    topic_modeler.save_model()
    network_analyzer.save_network(network_data)
    save_processed_data(df)
    
    logger.info("Pipeline execution complete.")

if __name__ == "__main__":
    main()