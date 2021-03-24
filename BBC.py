import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import nltk
from string import digits
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
lemmatiser = WordNetLemmatizer()
from sklearn import svm
#!pip install newspaper3k
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split

#pre-process text to remove short words
def filterLen(tdocs, minlen):
    return [ ' '.join(t for t in d if len(t) >= minlen ) for d in tdocs ]

#remove stop-words, tokenize and lemmatize the text
def lemmatize(train_filtered):
    for t in train_filtered:
        word_tokens = word_tokenize(t[0]) 
        filtered_sentence = [lemmatiser.lemmatize(w) for w in word_tokens if not w in stop_words] 
        t = filtered_sentence
    return train_filtered
    

def scrape_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def process_url(url):
    df_in = pd.read_csv('bbc-text.csv', encoding='latin1')
    df_in.head()
    trainDocs = [l.split() for l in df_in['text']]
    train_filtered = filterLen(trainDocs,4)
    df_filteredTxt = lemmatize(train_filtered)
    df_filteredTxt = pd.DataFrame(train_filtered)
    df_filteredTxt.columns =['text']
    #pre-processed dataframe
    df_p1 = df_in['category']
    df_p2 = df_filteredTxt
    df = pd.concat([df_p1,df_p2],axis = 1)

    df['category_id'] = df['category'].factorize()[0]
    #dataframe with just the ids and corresponding category
    category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
 
    #convert to a dictionary
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    #pre-process


    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    features = tfidf.fit_transform(df.text).toarray()
    labels = df.category_id


    for category, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]



    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels,
                                                                                 df.index, test_size=0.25, random_state=0)


    svm_model = svm.LinearSVC()
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    print(sklearn.metrics.accuracy_score(y_test,y_pred))
    texts = []
    texts.append(scrape_text(url))

    #pre-process input
    testDoc = [l.split() for l in texts]
    test_filtered = filterLen(testDoc,4)
    test_lemma = lemmatize(test_filtered)
    text_features = tfidf.transform(test_lemma)

    #predict
    predictions = svm_model.predict(text_features)
    #print(texts,id_to_category[predictions])

    for text, predicted in zip(texts, predictions):
        return text,id_to_category[predicted]
        '''print('"{}"'.format(text))
        print("  - Predicted as: '{}'".format(id_to_category[predicted]))
        print("**********************")'''




process_url('https://www.cnn.com/2018/12/01/politics/george-h-w-bush-dead/index.html')


#urls = ['https://www.cnn.com/2018/12/01/politics/george-h-w-bush-dead/index.html',
#        'https://www.theverge.com/2018/12/1/18121170/fox-national-geographic-neil-degrasse-tyson-sexual-misconduct-claims-investigation']
