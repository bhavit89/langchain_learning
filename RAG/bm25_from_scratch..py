import math 
from collections import Counter 

class BM25:
    def __init__(self ,documents):
        self.documents = documents
        self.document_count = len(documents)

        # calculate the average document length 
        self.avaerage_document_length = sum(len(doc) for doc in documents)/self.document_count
        

        # Calculate document frequency and term_document_frequency 
        "document frequency - Number of documents that contains the unique word"
        "term frequenct -  Number of times the terms appears in a specific document"

        self.doc_freq = Counter()
        self.term_freq = Counter()

        for document in  documents:
            self.doc_freq.update(set(document))
            self.term_freq.update(document)

    def calculate_idf(self, term):
        """
        Term Frequency - Number of times term appear in document d/Total number of terms in document d
        IDF def - log(Total number of documents in corpus D)/Number of documents containig term t
        """

        # Number of documents containing the term
        n_q = self.doc_freq.get(term, 0)

        # Bm25-formula
        # log((N - n_q + 0.5) / (n_q + 0.5))
        # where N is total number of documents

        # To avoid zero division by zero or negative values 
        if n_q == 0:
          return 0
        # Use positive IDF formula
        return math.log(1 + (self.document_count - n_q + 0.5) / (n_q + 0.5))
        
    def calculate_bm25_scores(self ,query ,document_index):
        """ Calculate BM25 scores for a query-document pair"""
        document = self.documents[document_index]
        document_length = len(document)
        score = 0.0

        doc_term_freq = Counter(document)
        for term in set(query):

            if term not in doc_term_freq:
                continue
            
            # term frequency in the document
            f_qi = doc_term_freq[term]
            idf = self.calculate_idf(term)

            numerator = f_qi * (self.k1 + 1)
            denominator = f_qi + self.k1 * (1 - self.b + self.b * (document_length / self.avaerage_document_length))

            if denominator > 0:
                score += idf * (numerator/denominator)
            
        return score
        

    def rank_document(self , query):
        """ Rank  all document based on the  BM25 query"""
        self.k1 = 1.2 # controls term frerquency  range 1.2 - 2.0
        self.b = 0.75 # control documetn lenght normalization 0.5 -0.8

        document_scores = []

        for i , document in enumerate(self.documents):
            score  = self.calculate_bm25_scores(query, i)
            document_scores.append((i,document,score))

        # Sort documents by score in descending order
        ranked_documents = sorted(document_scores, key=lambda x: x[2], reverse=True)
        return ranked_documents

def preprocess_text(text):
    """
    simple pre-processing of the document
    """
    return text.lower().split()





D1 = "London is windy today"
D2 = "It is quite windy"
D3 = "The weather is nice in Paris"
query = "windy London"

d1 = preprocess_text(D1)
d2 = preprocess_text(D2)
d3 = preprocess_text(D3)

documents = [d1 , d2, d3]
query_tokens = preprocess_text(query)

bm25 = BM25(documents)
ranked_docs = bm25.rank_document(query_tokens)

print("Query:", query)
print("\nRanked documents:")
for rank, (doc_idx, doc_tokens, score) in enumerate(ranked_docs, 1):
    original_doc = [D1, D2, D3][doc_idx]
    print(f"{rank}. Document {doc_idx + 1}: Score = {score:.4f}")
    print(f"   Text: '{original_doc}'")
    print(f"   Tokens: {doc_tokens}")
    print()