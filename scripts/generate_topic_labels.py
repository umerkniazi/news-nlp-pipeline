from src.topic_modeler import TopicModeler

tm = TopicModeler()
tm.load_model()

descriptors = tm.get_topic_descriptors(num_words=15)

prompt = (
    "You are an expert news editor. I have applied topic modeling to a dataset of "
    "350,000 news headlines from Dawn News, a major Pakistani newspaper covering "
    "2010-2025. Below are the latent topics extracted, represented by their most "
    "dominant keywords.\n\n"
    "For each topic, provide a concise, 1-to-3 word geopolitical or news category label. "
    "Output the result strictly as a valid JSON dictionary where the keys are the "
    "topic IDs (as strings) and the values are your generated labels. Do not include "
    "any other text or markdown formatting.\n\n"
    "Topics:\n"
)

for topic_id, keywords in descriptors.items():
    prompt += f'Topic {topic_id}: {", ".join(keywords)}\n'

print(prompt)