import pandas as pd
from nltk.corpus import gutenberg
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from tidytext import bind_tf_idf
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import tomotopy as tp
import nltk
from sklearn.decomposition import TruncatedSVD
from scipy.stats.contingency import association


#--------Load the books--------#


#print(gutenberg.fileids())
book_ids = ['austen-emma.txt',
            'austen-sense.txt',
            'shakespeare-caesar.txt',
            'shakespeare-hamlet.txt',
            'shakespeare-macbeth.txt',
            'melville-moby_dick.txt'
            ]

book_corpus = []
for id in book_ids:
    book_corpus.append(gutenberg.raw(id))

 
#--------Structuring and tidying--------#


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer() 

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())  
    return normalized

clean_corpus = [clean(doc).split() for doc in book_corpus]

titles = []
for doc in book_corpus:
    titles.append(doc[1:doc.find(']')])

clean_corpus_df = pd.DataFrame(columns=['title','word'])
counter = 0
for i, doc in enumerate(clean_corpus):
    for word in doc:
        clean_corpus_df.loc[counter,'title'] = titles[i]
        clean_corpus_df.loc[counter,'word'] = word
        counter += 1
print(clean_corpus_df.head(10))


#--------Term frequency and tf-idf--------#


frequency_df = pd.DataFrame({'n': clean_corpus_df.groupby(['title', 'word']).size().sort_values(ascending=False)}).reset_index()

tf_idf = bind_tf_idf(frequency_df,'word','title','n').sort_values(by='tf_idf', ascending=False)
print(frequency_df.head(10))
print(tf_idf[['title','word','tf_idf']].head(10))


#--------Visualisation--------#


def viz_barplots(tidy_df, freq):
    viz_df=pd.DataFrame(columns=['title','word',freq])
    for title in titles:
        top_words = (tidy_df[tidy_df['title'] == title].nlargest(10, freq))
        viz_df = pd.concat(objs=[viz_df,top_words],ignore_index=True)

    # Form a facetgrid using columns
    sns.set_style('whitegrid')
    f = sns.FacetGrid(viz_df, col='title', col_wrap=2, sharey=(False), height=3, aspect=1.7) 
    f.map(sns.barplot, freq, 'word', color = '#0078cb')

viz_barplots(tf_idf[['title','word','tf_idf']], 'tf_idf')


#--------Document-term matrices--------#


#---text to dtm using tf-idf
tf_idf_vectorizer = TfidfVectorizer(stop_words=stop, smooth_idf=False)
tf_idf_dtm = tf_idf_vectorizer.fit_transform(book_corpus)


#--------N-grams and analyzing n-grams--------#


#tf-idf bigrams
tf_idf_bigram_vec = TfidfVectorizer(stop_words=list(stop), smooth_idf=False, ngram_range=(2,2))
tf_idf_bigram_dtm = tf_idf_bigram_vec.fit_transform(book_corpus)

bigram_tf_idf_dtm_df = pd.DataFrame(tf_idf_bigram_dtm.toarray(), columns=tf_idf_bigram_vec.get_feature_names_out(), index=titles)
tidy_bigram_tf_idf_df = (bigram_tf_idf_dtm_df
                  .stack()
                  .reset_index(name='tf_idf')
                  .rename(columns={'level_0':'title','level_1':'word'})
                  .sort_values(by = 'tf_idf', ascending = False))
print(tidy_bigram_tf_idf_df)

#Visualize bigrams
viz_barplots(tidy_bigram_tf_idf_df, 'tf_idf')


#--------Latent Dirichlet Allocation--------#


lda = LatentDirichletAllocation(n_components=6, max_iter=20, random_state=42)

X_topics = lda.fit_transform(tf_idf_dtm)
topic_words = lda.components_
vocab_tf_idf = tf_idf_vectorizer.get_feature_names_out()

lda_topic_words_df = pd.DataFrame(columns=['Topic','Word'])
index, n_top_words = 0, 11
for i, topic_dist in enumerate(topic_words):
    sorted_topic_dist = np.argsort(topic_dist)
    topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
    topic_words = topic_words[:-n_top_words:-1]
    for word in topic_words:
        lda_topic_words_df.loc[index,'Topic'] = i
        lda_topic_words_df.loc[index,'Word'] = word
        index += 1
print(lda_topic_words_df)
    
topic_results = lda.transform(tf_idf_dtm)
topics_df = pd.DataFrame(data=titles, columns=['title'])
topics_df['Topic_LDA'] = topic_results.argmax(axis=1)
print(topics_df)

lda_gamma_df = pd.DataFrame(columns=['Text','Topic','Probability'])
text, index = 0, 0
for texts in topic_results:
    lda_gamma_df.loc[index,'Text'] = text
    lda_gamma_df.loc[index,'Topic'] = list(texts).index(max(abs(texts)))
    lda_gamma_df.loc[index,'Probability'] = max(abs(texts)).round(2)
    index += 1
    text += 1    
print(lda_gamma_df)


#--------Correlated Topic Model--------#


porter_stemmer = nltk.PorterStemmer().stem
eng_stops = set(porter_stemmer(w) for w in stopwords.words('english'))
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
    stopwords=lambda x: x in eng_stops or len(x) <= 2
)
corpus.process(book_corpus)

ctm = tp.CTModel(tw=tp.TermWeight.IDF, k=6, seed=42, corpus=corpus)
ctm.train(20)

doc_topic_dists = [doc.get_topics(top_n=1) for doc in ctm.docs]
ctm_gamma_df = pd.DataFrame(columns=['Text','Topic','Probability'])
index, topic =  0, 0
for doc in doc_topic_dists:
    for data in doc:
        ctm_gamma_df.loc[index,'Text'] = index
        ctm_gamma_df.loc[index,'Topic'] = data[0]
        ctm_gamma_df.loc[index,'Probability'] = round(data[1],2)
        topic += 1
        index += 1
print(ctm_gamma_df)

ctm_topic_words_df = pd.DataFrame(columns=['Topic','Word'])
i = 0
for k in range(ctm.k):
    for word, _ in ctm.get_topic_words(k, top_n=10):
        ctm_topic_words_df.loc[i,'Topic'] = k
        ctm_topic_words_df.loc[i,'Word'] = word
        i += 1
print(ctm_topic_words_df)


#--------Latent Semantic Analysis--------#


lsa = TruncatedSVD(n_components=6, n_iter=20, random_state=42)
lsa_data = lsa.fit_transform(tf_idf_dtm)

terms = tf_idf_vectorizer.get_feature_names_out()
lsa_topic_words_df = pd.DataFrame(columns=['Topic','Word'])
i = 0
for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
    top_terms_list = list(dict(top_terms_key).keys())
    for word in top_terms_list:
        lsa_topic_words_df.loc[i,'Topic'] = index
        lsa_topic_words_df.loc[i,'Word'] = word
        i += 1
print(lsa_topic_words_df)        

lsa_topic_results = lsa.transform(tf_idf_dtm)
lsa_gamma_df = pd.DataFrame(columns=['Text','Topic','Probability'])
text, index = 0, 0
for texts in lsa_topic_results:
    lsa_gamma_df.loc[index,'Text'] = text
    lsa_gamma_df.loc[index,'Topic'] = list(texts).index(max(abs(texts)))
    lsa_gamma_df.loc[index,'Probability'] = max(abs(texts)).round(2)
    index += 1
    text += 1
print(lsa_gamma_df)


#--------Cramer's V--------#


#Data prep
text_topics_df = lda_gamma_df[['Text','Topic']]
text_topics_df = text_topics_df.rename(columns={'Text':'Text','Topic':'LDA_Topic'})
text_topics_df['CTM_Topic'] = ctm_gamma_df.iloc[:,1]
text_topics_df['LSA_Topic'] = lsa_gamma_df.iloc[:,1]
print(text_topics_df)

#Contingency table
LDA_CTM_crosstab = pd.crosstab(text_topics_df['LDA_Topic'],text_topics_df['CTM_Topic'])
LDA_LSA_crosstab = pd.crosstab(text_topics_df['LDA_Topic'],text_topics_df['LSA_Topic'])
CTM_LSA_crosstab = pd.crosstab(text_topics_df['CTM_Topic'],text_topics_df['LSA_Topic'])
print(LDA_CTM_crosstab,'\n', LDA_LSA_crosstab,'\n', CTM_LSA_crosstab)

#CramerV
LDA_CTM_cramer = round(association(LDA_CTM_crosstab, 'cramer'),2)
LDA_LSA_cramer = round(association(LDA_LSA_crosstab, 'cramer'),2)
CTM_LSA_cramer = round(association(CTM_LSA_crosstab, 'cramer'),2)
print(LDA_CTM_cramer,'\n', LDA_LSA_cramer,'\n', CTM_LSA_cramer)