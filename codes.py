import pandas as pd
from tidytext import unnest_tokens, bind_tf_idf
from nltk.corpus import stopwords
import string

path = 'C:\\Users\zsolt\Desktop\Szakszem\Szakdoga\OV_speeches.csv'
new_dataset_df = pd.read_csv(path, encoding='latin1', sep=';')

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


#--------Term frequency and tf-idf--------#
frequency_df = pd.DataFrame({'n': tidy_df.groupby(['title', 'word']).size().sort_values(ascending=False)}).reset_index()

tf_idf = bind_tf_idf(frequency_df,'word','title','n').sort_values(by='tf_idf', ascending=False)
#print(tf_idf[['title','word','tf_idf']].head(15))


#--------N-grams and correlations--------#

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
            df1 = {'title': title, 'bigram': word}
            bigram_df = bigram_df.append(df1, ignore_index = True)

#---tf and tf-idf for bigrams
tf_bigram_df = pd.DataFrame({'n': bigram_df.groupby(['title', 'bigram']).size().sort_values(ascending=False)}).reset_index()

tf_idf_bigram = bind_tf_idf(tf_bigram_df,'bigram','title','n').sort_values(by='tf_idf', ascending=False) #all idf 0 -- check later
#print(tf_idf_bigram[['title','bigram','idf']].head(15))























