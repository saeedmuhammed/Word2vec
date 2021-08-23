import os

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn import svm
import warnings
warnings.filterwarnings('ignore')  # Warning people, manual storage
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec

StopWords = stopwords.words("english")


def get_sentence_vec(sentence,model):
    word_vectors = model.wv
    words = sentence.split()
    final=[]
    for word in words:
       if word in word_vectors.key_to_index:
            final.append(word)

    return np.mean(model.wv[final], axis=0)


def proccess_text(text):
    text = ''.join([character for character in text if not character.isdigit()])  # remove numbers

    text = ''.join(character for character in text if
                   character.isalnum() or character == ' ')  # remove special charactars ==' ' 3shan kan byms7 el space ely ben el kalemat
    text = " ".join(text.split())  # remove extra spaces

    text = ' '.join([word for word in text.split() if word not in StopWords])
    return text


posetive_path = "D:\\Universty\\Last year\\2d Term\\NLP\\txt_sentoken\\pos"
negative_path = "D:\\Universty\\Last year\\2d Term\\NLP\\txt_sentoken\\neg"


reviews_words = []
reviews=[]

for root, dirs, files in os.walk(posetive_path):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                temp=[]
                text = proccess_text(text)
                reviews.append(text)
                for word in text.split():
                    temp.append(word)
                reviews_words.append(temp)
for root, dirs, files in os.walk(negative_path):
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                text = proccess_text(text)
                reviews.append(text)
                for word in text.split():
                    temp.append(word)
                reviews_words.append(temp)


#model=Word2Vec(sentences=reviews_words,vector_size=2000,window=10,min_count=4,workers=4,epochs=60)
#model.save("word2vec2.model")

model = Word2Vec.load("word2vec2.model")
reviews_features=[]
for review in reviews:
    reviews_features.append(get_sentence_vec(review,model))
target = []
for i in range(0, 2000):
    if i < 1000:
        target.append(1)
    else:
        target.append(0)



X_train, X_test, y_train, y_test = train_test_split(reviews_features, target, test_size=0.2, random_state=33)





clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

print("Testing-Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training-Accuracy:",metrics.accuracy_score(y_train, y_pred_train))
#------------------------------------------------------------------

model = LogisticRegression(random_state=0).fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Training-accuracy:   %0.3f" % train_acc)

print("Testing-accuracy:   %0.3f" % test_acc)

#---------------------------------------------------------------

#for normalizing data and make the range between 0 , 1 and remove negative  numbers

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)


y_pred = naive_bayes_classifier.predict(X_test)
y_pred_train = naive_bayes_classifier.predict(X_train)


score1 = metrics.accuracy_score(y_test, y_pred)
print("Testing-accuracy:   %0.3f" % score1)


score2 = metrics.accuracy_score(y_train, y_pred_train)
print("Training-accuracy:   %0.3f" % score2)

