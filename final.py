import pandas as pd   
from bs4 import BeautifulSoup  
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

def review_to_words(raw_review):
    
    review_text = BeautifulSoup(raw_review,'lxml').get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]  
    return( '\n'," ".join( meaningful_words )) 

num_reviews = train["review"].size

clean_train_reviews = []

print("Cleaning and parsing the training set movie reviews ... \n")
clean_train_reviews = []
for i in range(0, num_reviews):
    if((i+1)%1000 == 0 ):
        print("Review %d of %d\n" %(i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

print("Creating the bag of words...\n")

vectorizer = CountVectorizer(analyzer = "word", \
                            tokenizer = lambda doc:doc, \
                            preprocessor = None, \
                            stop_words = None,
                            max_features = 5000,
                            lowercase=False)

train_data_features = vectorizer.fit_transform(clean_train_reviews)

#train_data_features = train_data_features.toarray()
np.asarray(train_data_features)

print (train_data_features.shape)

vocab = vectorizer.get_feature_names()
print(vocab)


dist = np.sum(train_data_features, axis=0)

for tag, count in zip(vocab, dist):
    print(count, tag)


forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit( train_data_features, train["sentiment"] )

print(""" testing part beings here """)


test = pd.read_csv("testData.tsv", header=0, delimiter="\t",quoting=3 )

print(test.shape)

num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning the test set \n")
for i in range(0, num_reviews):
    if((i+1) % 1000 == 0):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)

result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id":test["id"],"sentiment":result})

output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
print('A CSV file with the predictions has been created.')

