from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

max_features = 1000

def join(text):
    string =' '.join(map(str,text))
    return string

df['final'] = df['stemming'].apply(join)

# new_df  = pd.DataFrame(data = df[['final','sentiment']])
# new_df.to_csv("testing.csv")


def generate_tfidf_mat(min_gram, max_gram):
    cvect = CountVectorizer(max_features=max_features, ngram_range=(min_gram, max_gram))
    counts = cvect.fit_transform(df["final"])

    normalized_counts = normalize(counts, norm='l1', axis=1)

    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(min_gram, max_gram), smooth_idf=False)
    tfs = tfidf.fit_transform(df["final"])

    tfidf_mat = normalized_counts.multiply(tfidf.idf_).toarray()
    
    TF = normalized_counts.toarray()
    IDF = tfidf.idf_
    TF_IDF = tfidf_mat
    return TF, IDF, TF_IDF, tfidf.get_feature_names()


#test


# ngram_range (1, 1) to use unigram only
tf_mat_unigram, idf_mat_unigram, tfidf_mat_unigram, terms_unigram = generate_tfidf_mat(1,1)

# ngram_range (2, 2) to use bigram only
tf_mat_bigram, idf_mat_bigram, tfidf_mat_bigram, terms_bigram = generate_tfidf_mat(2,2)

# ngram_range (3, 3) to use trigram only
tf_mat_trigram, idf_mat_trigram, tfidf_mat_trigram, terms_trigram = generate_tfidf_mat(3,3)

# # ---------- check sparse data -------------------
# idx_sample = 0

# print("Show TFIDF sample ke-" + str(idx_sample), "\n")
# print(df["final"][idx_sample], "\n")

# print("\t\t\t", "TF", "\t\t", "IDF", "\t\t", "TF-IDF", "\t", "Term\n")
# for i, item in enumerate(zip(tf_mat_unigram[idx_sample], idf_mat_unigram, tfidf_mat_unigram[idx_sample], terms_unigram)):
#     if(item[2] != 0.0):
#         print ("array position " + str(i) + "\t", 
#                "%.6f" % item[0], "\t", 
#                "%.6f" % item[1], "\t", 
#                "%.6f" % item[2], "\t", 
#                item[3])
        
def get_TF_unigram(row):
    idx = row.name
    return [tf for tf in tf_mat_unigram[idx] if tf != 0.0]

df["TF_UNIGRAM"] = df.apply(get_TF_unigram, axis=1)

def get_IDF_unigram(row):
    idx = row.name
    return [item[1] for item in zip(tf_mat_unigram[idx], idf_mat_unigram) if item[0] != 0.0]

df["IDF_UNIGRAM"] = df.apply(get_IDF_unigram, axis=1)

def get_TFIDF_unigram(row):
    idx = row.name
    return [tfidf for tfidf in tfidf_mat_unigram[idx] if tfidf != 0.0]

df["TFIDF_UNIGRAM"] = df.apply(get_TFIDF_unigram, axis=1)

# df[["final", "TF_UNIGRAM", "IDF_UNIGRAM", "TFIDF_UNIGRAM", "sentiment"]]

# save TFIDF Unigram to Excel

# df[["final", "TF_UNIGRAM", "IDF_UNIGRAM", "TFIDF_UNIGRAM"]].to_csv("TFIDF_Unigram.csv")