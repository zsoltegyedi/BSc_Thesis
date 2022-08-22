import pandas as pd
from tidytext import unnest_tokens, bind_tf_idf
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

path = 'C:\\Users\zsolt\Desktop\Szakdoga\OV_speeches.csv'
path1 = 'C:\\Users\zsolt\Desktop\Szakdoga\OV_speeches1.csv'
new_dataset_df = pd.read_csv(path, encoding='utf-8', sep=';')


dataset = """
Isten, áldd meg a magyart
Jó kedvvel, bőséggel,
Nyújts feléje védő kart,
Ha küzd ellenséggel;
Bal sors akit régen tép,
Hozz rá víg esztendőt,
Megbünhödte már e nép
A multat s jövendőt!
Őseinket felhozád
Kárpát szent bércére,
Általad nyert szép hazát
Bendegúznak vére.
S merre zúgnak habjai
Tiszának, Dunának,
Árpád hős magzatjai
Felvirágozának."""


#--------Structuring and tidying--------#

dataset_split = dataset.splitlines()
dataset_df = pd.DataFrame({
    'data': dataset_split})

tidy_df = unnest_tokens(new_dataset_df, 'word', 'text')

#--------Stop words removal--------#

hungarian_stops = pd.array(stopwords.words('hungarian'))
stop_words_df = pd.DataFrame({'word':hungarian_stops})

merge_df = pd.merge(tidy_df,stop_words_df, how='outer', left_on='word', right_on='word', indicator = True)
dataset_without_stopwords = merge_df.loc[merge_df['_merge']=='left_only']
#print(dataset_without_stopwords)

#--------Term frequency and tf-idf--------#

frequency_df = pd.DataFrame({'n': tidy_df.groupby(['title', 'word']).size().sort_values(ascending=False)}).reset_index()

tf_idf = bind_tf_idf(frequency_df,'word','title','n').sort_values(by='tf_idf', ascending=False)
#print(tf_idf[['title','word','tf_idf']].head(15))


#--------N-grams and analyzing n-grams--------#

#---defining the function to remove punctuation
def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""  
  for i in text:     
    if i not in string.punctuation:
      ans+=i    
  return ans

#---defining the function to generate n-grams    
def generate_N_grams(text,ngram=1):
  words=[word for word in text.split(" ") if word not in hungarian_stops]
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

#---create a structured bigram-dataframe
bigram_df = pd.DataFrame(columns=['title','bigram'], index=[0])
for ind in new_dataset_df.index:
    title = new_dataset_df['title'][ind]
    for text in new_dataset_df.text:
        for word in generate_N_grams(remove_punctuation(text),2):
            df1 = pd.DataFrame({'title': title, 'bigram': word}, index=[0])
            bigram_df = pd.concat([bigram_df, df1], ignore_index = True)
#print(bigram_df)

#---tf and tf-idf for bigrams
tf_bigram_df = pd.DataFrame({'n': bigram_df.groupby(['title', 'bigram'])
                             .size()
                             .sort_values(ascending=False)}).reset_index()

tf_idf_bigram = bind_tf_idf(tf_bigram_df,'bigram','title','n').sort_values(by='tf_idf', ascending=False) #all idf 0 -- check later
#print(tf_idf_bigram[['title','bigram','tf_idf']].head(15))


#--------Document-term matrices and corpuses--------#

#---text to dtm
vec = CountVectorizer(stop_words=list(stopwords.words('hungarian')))
dtm = vec.fit_transform(new_dataset_df['text'])
dtm_df = pd.DataFrame(dtm.toarray(), columns=vec.get_feature_names_out(), index=new_dataset_df['title'])
#print(dtm_df)

#---text to dtm using tf-idf
tf_idf_vectorizer = TfidfVectorizer(stop_words=list(stopwords.words('hungarian')))
tf_idf_dtm = tf_idf_vectorizer.fit_transform(new_dataset_df['text'])

#---dtm to tidy text format
tidied_dtm_df = (dtm_df
                  .stack()
                  .reset_index(name='n')
                  .rename(columns={'level_1':'word'}))
#print(tidied_dtm_df)


#--------Latent Dirichlet Allocation--------#


lda = LatentDirichletAllocation(n_components=3, max_iter=20, random_state=20)

X_topics = lda.fit_transform(tf_idf_dtm)
topic_words = lda.components_
vocab_tf_idf = tf_idf_vectorizer.get_feature_names_out()

LDA_topic_words_df = pd.DataFrame(columns=['Topic','Word'])
index, n_top_words = 0, 10
for i, topic_dist in enumerate(topic_words):
    sorted_topic_dist = np.argsort(topic_dist)
    topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
    topic_words = topic_words[:-n_top_words:-1]
    for word in topic_words:
        LDA_topic_words_df.loc[index,'Topic'] = i
        LDA_topic_words_df.loc[index,'Word'] = word
        index += 1

print(LDA_topic_words_df)
    
topic_results = lda.transform(tf_idf_dtm)
new_dataset_df['Topic_tfidf'] = topic_results.argmax(axis=1)
print(new_dataset_df)

LDA_gamma_df = pd.DataFrame(columns=['Text','Topic','Probability'])
text, topic, index = 0, 0, 0
for texts in topic_results:
    for topics in texts:
        LDA_gamma_df.loc[index,'Text'] = text
        LDA_gamma_df.loc[index,'Topic'] = topic
        LDA_gamma_df.loc[index,'Probability'] = topics.round(2)
        topic += 1
        index += 1
    text += 1
    topic = 0
    

print(LDA_gamma_df)


#--------Correlated Topic Model--------#


import tomotopy as tp
import nltk

porter_stemmer = nltk.PorterStemmer().stem
hun_stops = set(porter_stemmer(w) for w in stopwords.words('hungarian'))
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
    stopwords=lambda x: x in hun_stops or len(x) <= 2
)
corpus.process(open('OV_speeches1.txt', encoding='utf-8'))



CTM = tp.CTModel(tw=tp.TermWeight.IDF, k=3, seed=20, corpus=corpus)
CTM.train(20)

doc_topic_dists = [doc.get_topics(top_n=1) for doc in CTM.docs]
CTM_gamma_df = pd.DataFrame(columns=['Text','Topic','Probability'])
index =  0
for doc in doc_topic_dists:
    for data in doc:
        CTM_gamma_df.loc[index,'Text'] = index
        CTM_gamma_df.loc[index,'Topic'] = data[0]
        CTM_gamma_df.loc[index,'Probability'] = data[1]
        topic += 1
        index += 1
print(CTM_gamma_df)

CTM_topic_words_df = pd.DataFrame(columns=['Topic','Word'])
i = 0
for k in range(CTM.k):
    for word, _ in CTM.get_topic_words(k, top_n=10):
        CTM_topic_words_df.loc[i,'Topic'] = k
        CTM_topic_words_df.loc[i,'Word'] = word
        i += 1
print(CTM_topic_words_df)


# VISUALIZATION FOR CTM
"""
from pyvis.network import Network

g = Network(width=800, height=800, font_color="#333")
correl = CTM.get_correlations().reshape([-1])
correl.sort()
top_tenth = CTM.k * (CTM.k - 1) // 10
top_tenth = correl[-CTM.k - top_tenth]


for k in range(CTM.k):
    label = "#{}".format(k)
    title= ' '.join(word for word, _ in CTM.get_topic_words(k, top_n=10))
    print('Topic', label, title)
    g.add_node(k, label=label, title=title, shape='ellipse')
    for l, correlation in zip(range(k - 1), CTM.get_correlations(k)):
        if correlation < top_tenth: continue
        g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))

g.barnes_hut(gravity=-1000, spring_length=20)
g.show_buttons()
g.show("topic_network.html")
"""














