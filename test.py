from sentence_transformers import SentenceTransformer

def read_markdown(filepath: str):
    """ read markdown and return sentences list[str] """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().split("\n")

if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sentences = read_markdown("test.md")
    embeddings = model.encode(sentences)
    print(embeddings)

"""
[[-0.25564343 -0.10067613 -0.02242378 ...  0.03095629 -0.04536473
  -0.01467749]
 [ 0.22691742  0.08178423  0.02354268 ... -0.09982924 -0.03107586
   0.0741839 ]
 [-0.1133671   0.05849995  0.02576483 ...  0.09931883 -0.07779369
   0.06022361]
 ...
 [-0.09429257 -0.004644   -0.1021011  ...  0.25961316 -0.16159451
  -0.11975738]
 [-0.09671228 -0.08585478 -0.10871892 ...  0.24735849 -0.23145473
  -0.04328802]
 [-0.24758902 -0.13027966 -0.04999169 ...  0.2626674  -0.21395649
  -0.09438755]]
"""