from src.topic_modeler import TopicModeler

def generate_topic_labels(num_words: int = 15):
    print("Loading trained LDA model...")
    tm = TopicModeler()
    tm.load_model()
    
    descriptors = tm.get_topic_descriptors(num_words=num_words)
    
    prompt = (
        "You are an expert news editor. I have applied topic modeling to a dataset of "
        "350,000 news headlines. Below are the latent topics extracted, represented by "
        "their most dominant keywords.\n\n"
        "For each topic, provide a concise, 1-to-3 word geopolitical or news category label. "
        "Output the result strictly as a valid JSON dictionary where the keys are the "
        "topic IDs (as strings) and the values are your generated labels. Do not include "
        "any other text or markdown formatting.\n\n"
        "Topics:\n"
    )
    
    for topic_id, keywords in descriptors.items():
        keyword_str = ", ".join(keywords)
        prompt += f'Topic {topic_id}: {keyword_str}\n'
        
    print("\n" + "=" * 80)
    print("COPY THE TEXT BELOW:")
    print("=" * 80 + "\n")
    print(prompt)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    generate_topic_labels()