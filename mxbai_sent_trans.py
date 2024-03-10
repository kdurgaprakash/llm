from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

prompt='Find relevant sentences:'
query = f'{prompt} Astronaut skateboarding inside space station. '

docs =  [
    query,
    "A kid is playing cricket.",
    "Astronaut is flying on a plane.",
    "Girl fell on a skateboard.",
    "A person is writing a diary.",
    "Dog is barking."
]


embeddings = model.encode(docs)

similarities = cos_sim(embeddings[0],embeddings[1:])
print('similarities:', similarities)

