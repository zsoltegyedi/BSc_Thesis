from tidytext import bind_tf_idf

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

#path = 'C:\\Users\zsolt\Desktop\Szakdoga\OV_speeches.csv'
#path1 = 'C:\\Users\zsolt\Desktop\Szakdoga\OV_speeches1.csv'
#new_dataset_df = pd.read_csv(path, encoding='utf-8', sep=';')



#--------Structuring and tidying--------#


#https://figshare.com/articles/dataset/Analyzing_Presidential_Speeches_with_Topic_Modeling/2060724/1?file=3660411
import os
import pandas as pd

root = 'C:\\Users\zsolt\Desktop\Szakdoga\obama_speeches'
files = os.listdir(root)

#load speeches into a list
docs_df = pd.DataFrame(columns=['title','text'])
docs = list()
i = 0
for file in files:
    with open(os.path.join(root, file), 'r') as fd:
       txt = fd.read()
       docs.append(txt)
       docs_df.loc[i,'title'] = file
       docs_df.loc[i,'text'] = txt
       i += 1
       
#tidy_df = unnest_tokens(docs_df, 'word', 'text')
#print(tidy_df)

# source for preprocessing: https://www.analyticsvidhya.com/blog/2021/06/part-3-topic-modeling-and-latent-dirichlet-allocation-lda-using-gensim-and-sklearn/
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) 
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())  
    return normalized

clean_corpus = [clean(doc).split() for doc in docs]
#print(clean_corpus)

clean_corpus_df = pd.DataFrame(columns=['title','word'])
counter = 0
for i, doc in enumerate(clean_corpus):
    for word in doc:
        clean_corpus_df.loc[counter,'title'] = files[i]
        clean_corpus_df.loc[counter,'word'] = word
        counter += 1


#--------Stop words removal--------#


#stop_words_df = pd.DataFrame({'word':stop})

#merge_df = pd.merge(clean_corpus_df,stop_words_df, how='outer', left_on='word', right_on='word', indicator = True)
#dataset_without_stopwords = merge_df.loc[merge_df['_merge']=='left_only']
#print(dataset_without_stopwords)

#--------Term frequency and tf-idf--------#

frequency_df = pd.DataFrame({'n': clean_corpus_df.groupby(['title', 'word'])
                             .size()
                             .sort_values(ascending=False)}).reset_index()
print(frequency_df)

tf_idf = bind_tf_idf(frequency_df,'word','title','n').sort_values(by='tf_idf', ascending=False)
print(tf_idf[['title','word','tf_idf']].head(15))


#--------Visualisation--------#

#https://www.geeksforgeeks.org/multi-plot-grid-in-seaborn/

import seaborn as sns

def viz_barplots(tidy_df, freq):
    viz_df=pd.DataFrame(columns=['title','word',freq])
    titles = files
    for title in titles:
        top_words = (tidy_df[tidy_df['title'] == title].nlargest(10, freq))
        viz_df = pd.concat(objs=[viz_df,top_words],ignore_index=True)
    # Form a facetgrid using columns
    sns.set_style('whitegrid')
    g = sns.FacetGrid(viz_df, col='title', col_wrap=3, sharey=(False), height=4) 
    g.map(sns.barplot, freq, 'word', color = '#0078cb')

viz_barplots(frequency_df, 'n')
viz_barplots(tf_idf[['title','word','tf_idf']], 'tf_idf')


#--------Document-term matrices--------#

#---text to dtm
vec = CountVectorizer(stop_words=list(stop))
tf_dtm = vec.fit_transform(docs_df['text'])

#---text to dtm using tf-idf
tf_idf_vectorizer = TfidfVectorizer(stop_words=list(stop), smooth_idf=False)
tf_idf_dtm = tf_idf_vectorizer.fit_transform(docs_df['text'])

#---dtm to tidy data format
tf_dtm_df = pd.DataFrame(tf_dtm.toarray(), columns=vec.get_feature_names_out(), index=docs_df['title'])
print(tf_dtm_df)
tidy_tf_df = (tf_dtm_df
                  .stack()
                  .reset_index(name='n')
                  .rename(columns={'level_1':'word'})
                  .sort_values(by = 'n', ascending = False))
print(tidy_tf_df)


#--------N-grams and analyzing n-grams--------#

#count bigrams
count_bigram_vec = CountVectorizer(stop_words=list(stop), ngram_range=(2,2))
count_bigram_dtm = count_bigram_vec.fit_transform(docs_df['text'])

bigram_tf_dtm_df = pd.DataFrame(count_bigram_dtm.toarray(), columns=count_bigram_vec.get_feature_names_out(), index=docs_df['title'])
tidy_bigram_tf_df = (bigram_tf_dtm_df
                  .stack()
                  .reset_index(name='n')
                  .rename(columns={'level_1':'word'})
                  .sort_values(by = 'n', ascending = False))
print(tidy_bigram_tf_df)

#tf-idf bigrams
tf_idf_bigram_vec = TfidfVectorizer(stop_words=list(stop), smooth_idf=False, ngram_range=(2,2))
tf_idf_bigram_dtm = tf_idf_bigram_vec.fit_transform(docs_df['text'])

bigram_tf_idf_dtm_df = pd.DataFrame(tf_idf_bigram_dtm.toarray(), columns=tf_idf_bigram_vec.get_feature_names_out(), index=docs_df['title'])
tidy_bigram_tf_idf_df = (bigram_tf_idf_dtm_df
                  .stack()
                  .reset_index(name='tf_idf')
                  .rename(columns={'level_1':'word'})
                  .sort_values(by = 'tf_idf', ascending = False))
print(tidy_bigram_tf_idf_df)

#Visualize bigrams
viz_barplots(tidy_bigram_tf_df, 'n')
viz_barplots(tidy_bigram_tf_idf_df, 'tf_idf')



#--------Latent Dirichlet Allocation--------#


lda = LatentDirichletAllocation(n_components=3, max_iter=20, random_state=20)

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
docs_df['Topic_tfidf'] = topic_results.argmax(axis=1)
print(docs_df)

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


import tomotopy as tp
import nltk

porter_stemmer = nltk.PorterStemmer().stem
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
    stopwords=lambda x: x in stop or len(x) <= 2
)

for file in files:
    corpus.process(open(os.path.join(root, file), 'r'))
#corpus.process(open('C:\\Users\zsolt\Desktop\Szakdoga\OV_speeches1.txt', encoding='utf-8'))



ctm = tp.CTModel(tw=tp.TermWeight.IDF, k=3, seed=20, corpus=corpus)
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


# VISUALIZATION FOR CTM
"""
from pyvis.network import Network

g = Network(width=800, height=800, font_color="#333")
correl = ctm.get_correlations().reshape([-1])
correl.sort()
top_tenth = CTM.k * (ctm.k - 1) // 10
top_tenth = correl[-ctm.k - top_tenth]


for k in range(ctm.k):
    label = "#{}".format(k)
    title= ' '.join(word for word, _ in ctm.get_topic_words(k, top_n=10))
    print('Topic', label, title)
    g.add_node(k, label=label, title=title, shape='ellipse')
    for l, correlation in zip(range(k - 1), ctm.get_correlations(k)):
        if correlation < top_tenth: continue
        g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))

g.barnes_hut(gravity=-1000, spring_length=20)
g.show_buttons()
g.show("topic_network.html")
"""


#--------Latent Semantic Analysis--------#

#https://machinelearninggeek.com/latent-semantic-indexing-using-scikit-learn/

from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=3, n_iter=20, random_state=20)

lsa_data = lsa.fit_transform(tf_idf_dtm)

# Print the topics with their terms
terms = tf_idf_vectorizer.get_feature_names_out()
lsa_topic_words_df = pd.DataFrame(columns=['Topic','Word'])
i = 0
for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
    top_terms_list = list(dict(top_terms_key).keys())
    #print("Topic "+str(index)+": ",top_terms_list)
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

#Visualisation of sigmas

"""
Sigma = lsa.singular_values_

import seaborn as sns
sns.barplot(x=list(range(len(Sigma))), y = Sigma)
"""


#--------Cramer's V--------#

#Data prep
text_topics_df = lda_gamma_df[['Text','Topic']]
text_topics_df = text_topics_df.rename(columns={'Text':'Text','Topic':'LDA_Topic'})
text_topics_df['CTM_Topic'] = ctm_gamma_df.iloc[:,1]
text_topics_df['LSA_Topic'] = lsa_gamma_df.iloc[:,1]
print(text_topics_df)

#Contingency tables
LDA_CTM_crosstab = pd.crosstab(text_topics_df['LDA_Topic'],text_topics_df['CTM_Topic'])
LDA_LSA_crosstab = pd.crosstab(text_topics_df['LDA_Topic'],text_topics_df['LSA_Topic'])
CTM_LSA_crosstab = pd.crosstab(text_topics_df['CTM_Topic'],text_topics_df['LSA_Topic'])
print(LDA_CTM_crosstab,'\n', LDA_LSA_crosstab,'\n', CTM_LSA_crosstab)

#Cramer's V

from scipy.stats.contingency import association

LDA_CTM_cramer = round(association(LDA_CTM_crosstab, 'cramer'),2)
LDA_LSA_cramer = round(association(LDA_LSA_crosstab, 'cramer'),2)
CTM_LSA_cramer = round(association(CTM_LSA_crosstab, 'cramer'),2)
print(LDA_CTM_cramer,'\n', LDA_LSA_cramer,'\n', CTM_LSA_cramer)



