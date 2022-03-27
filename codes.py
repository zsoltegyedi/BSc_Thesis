import pandas as pd
from tidytext import unnest_tokens
from nltk.corpus import stopwords


himnusz = """
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

himnusz_split = himnusz.splitlines()

himnusz_df = pd.DataFrame({
    "himnusz": himnusz_split,
    "line": list(range(len(himnusz_split)))})

tidy_himnusz = unnest_tokens(himnusz_df, "word", "himnusz")
#print(tidy_himnusz)


hungarian_stops = pd.array(stopwords.words('hungarian'))
stop_words_df = pd.DataFrame({"word":hungarian_stops})

merge_df = pd.merge(tidy_himnusz,stop_words_df, how='outer', left_on='word', right_on='word', indicator = True)
himnusz_without_stopwords = merge_df.loc[merge_df['_merge']=='left_only']
print(himnusz_without_stopwords)