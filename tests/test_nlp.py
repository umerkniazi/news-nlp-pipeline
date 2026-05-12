import pandas as pd
from src.nlp_processor import NLPProcessor

def test_sentiment_analyzer() -> None:
    """Verifies sentiment threshold logic for positive, negative, and neutral text."""
    processor = NLPProcessor()
    
    pos_score, pos_label = processor.analyze_sentiment("This is an incredible and wonderful breakthrough!")
    assert pos_label == "positive"
    assert pos_score > 0.0

    neg_score, neg_label = processor.analyze_sentiment("This is a terrible, awful, and devastating disaster.")
    assert neg_label == "negative"
    assert neg_score < 0.0

    neu_score, neu_label = processor.analyze_sentiment("The box is located on the table.")
    assert neu_label == "neutral"

def test_process_dataframe() -> None:
    """Verifies that the dataframe processing pipeline adds the correct columns."""
    processor = NLPProcessor()
    df_in = pd.DataFrame({'headline': ["Apple CEO Tim Cook visits Paris."]})
    
    df_out = processor.process_dataframe(df_in, text_column='headline')
    
    assert 'extracted_entities' in df_out.columns
    assert 'sentiment_score' in df_out.columns
    assert 'sentiment_label' in df_out.columns
    
    entities = df_out.iloc[0]['extracted_entities']
    assert len(entities) > 0