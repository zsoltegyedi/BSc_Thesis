import pandas as pd
from tidytext import unnest_tokens, bind_tf_idf
from nltk.corpus import stopwords

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
    "data": dataset_split})

tidy_df = unnest_tokens(new_dataset_df, "word", "text")
#print(tidy_df)


#--------Stop words removal--------#
hungarian_stops = pd.array(stopwords.words('hungarian'))
stop_words_df = pd.DataFrame({"word":hungarian_stops})

merge_df = pd.merge(tidy_df,stop_words_df, how='outer', left_on='word', right_on='word', indicator = True)
dataset_without_stopwords = merge_df.loc[merge_df['_merge']=='left_only']
#print(merge_df)


#--------Term frequency and tf-idf--------#
frequency_df = pd.DataFrame({'n': tidy_df.groupby(['title', 'word']).size().sort_values(ascending=False)}).reset_index()
print(frequency_df.head(15))

tf_idf = bind_tf_idf(frequency_df,'word','title','n').sort_values(by='tf_idf', ascending=False)
print(tf_idf[['title','word','tf_idf']].head(15))