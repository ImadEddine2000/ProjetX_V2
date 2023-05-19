import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('english'))
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def removeSpeCara(s:str):
    return re.sub(r"[^a-zA-Z]", "", s)

#retirer les url
def remove_url(s:str)->str:
    url_pattern = re.compile(r"http?://\S+|https?://\S+|www\.\S+|//S+")
    return url_pattern.sub("r", s)

#retirer les html
def remove_html(s:str)->str:
    html_pattern = re.compile(r"<.*?>")
    return html_pattern.sub("r", s)

# retirer les emojies
def remove_emoji(s:str)->str:
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002500-\U00002BEF"  
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
                      "]+", re.UNICODE)
    return emoji_pattern.sub("r", s)

def clean_and_lemmatize_string(s:str):
    l = []
    s_ = " ".join([remove_html(remove_url(word)) for word in s.split()])
    for word in word_tokenize(s_):
        word_ = removeSpeCara((remove_emoji(word)))
        if not word_ in stop_words:
            l.append(word_.lower())
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lemma_function = WordNetLemmatizer()
    return " ".join([lemma_function.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(l)])