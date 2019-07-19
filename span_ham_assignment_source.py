import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import np #pip install np

"""
논문명: 박찬준, 류법모. (2018). 기계학습 알고리즘 앙상블 기법을 이용한 Spam/Ham 분류. 한국정보과학회 학술발표논문집, (), 2043-2045.
Paper: Spam/Ham Classification using Ensemble Technique of Machine Learning Algorithms (Park Chanjun)
https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07503585

자연언어처리의 기초인 분류 문제에 대한 소스코드입니다.
NLTK를 이용하여 전처리를 진행한 후 5가지의 기계학습 알고리즘을 앙상블하여 스팸,햄 분류를 진행합니다.
자세한 내용은 논문을 참조해주세요.

실행시 주의사항: SMSSpamCollection의 경로를 지정해주세요.

"""
smsdata = open('SMSSpamCollection',encoding='utf8') #PATH SETTING


def preprocessing(text):   #Preprocessing
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] 
   
    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]
    
    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]
    
    # lower capitalization
    tokens = [word.lower() for word in tokens]
    
    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]

    preprocessed_text= ' '.join(tokens)
    return preprocessed_text



sms_data = []
sms_labels = []
cnt = 0
sencsv_reader = csv.reader(smsdata,delimiter='\t')
for line in sencsv_reader:
    # adding the sms_id
    sms_labels.append(line[0])
    sms_data.append(preprocessing(line[1]))

smsdata.close()


trainset_size = int(round(len(sms_data)*0.70))  #Split Train data and Test data
print('The training set size for this classifier is ' + str(trainset_size) + '\n')
x_train = np.array([''.join(el) for el in sms_data[0:trainset_size]])
y_train = np.array([el for el in sms_labels[0:trainset_size]])
x_test = np.array([''.join(el) for el in sms_data[trainset_size+1:len(sms_data)]])
y_test = np.array(([el for el in sms_labels[trainset_size+1:len(sms_labels)]]) or el in sms_labels[trainset_size+1:len(sms_labels)])

# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', strip_accents='unicode', norm='l2')
X_train = vectorizer2.fit_transform(x_train)
X_test = vectorizer2.transform(x_test)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf_NB = MultinomialNB().fit(X_train, y_train)
y_predicted_NB = clf_NB.predict(X_test)

# Decision tree
from sklearn import tree
clf_DT = tree.DecisionTreeClassifier().fit(X_train.toarray(), y_train)
y_predicted_DT = clf_DT.predict(X_test.toarray())

# Stochastic gradient descent
from sklearn.linear_model import SGDClassifier
#clf_SGD = SGDClassifier(alpha=.0001, n_iter=50).fit(X_train, y_train)
clf_SGD = SGDClassifier(alpha=.0001).fit(X_train, y_train)
y_predicted_SGD = clf_SGD.predict(X_test)

# Support Vector Machines
from sklearn.svm import LinearSVC
clf_SVM = LinearSVC().fit(X_train, y_train)
y_predicted_SVM = clf_SVM.predict(X_test)

# The Random forest algorithm
from sklearn.ensemble import RandomForestClassifier
clf_RFA = RandomForestClassifier(n_estimators=10)
clf_RFA.fit(X_train, y_train)
y_predicted_RFA = clf_RFA.predict(X_test)

#Model Ensemble
y_predicted_all=[[],[],[],[],[]] 
y_predicted_all[0]=y_predicted_NB
y_predicted_all[1]=y_predicted_DT
y_predicted_all[2]=y_predicted_SGD
y_predicted_all[3]=y_predicted_SVM
y_predicted_all[4]=y_predicted_RFA

y_predicted_all_max=[]

ham_cnt=0
spam_cnt=0

for i in range(0,len(y_predicted_NB)): 
    ham_cnt=0 
    spam_cnt=0
    for j in range(0,5): #5 Machine Learning Algorithm
        if(y_predicted_all[j][i]=='ham'): #If ham
            ham_cnt=ham_cnt+1 #ham ++
        else:
            spam_cnt=spam_cnt+1 #spam ++

    if(ham_cnt>=spam_cnt):  !
        y_predicted_all_max.append('ham') 
    else:
        y_predicted_all_max.append('spam')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print (' \n confusion_matrix y-predicted_all_max \n ')
cm = confusion_matrix(y_test, y_predicted_all_max)
print (cm)
print ('\n Here is the classification report:')
print (classification_report(y_test, y_predicted_all_max))



from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print (' \n confusion_matrix NB \n ')
cm = confusion_matrix(y_test, y_predicted_NB)
print (cm)
print ('\n Here is the classification report:')
print (classification_report(y_test, y_predicted_NB))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print (' \n confusion_matrix DT \n ')
cm = confusion_matrix(y_test, y_predicted_DT)
print (cm)
print ('\n Here is the classification report:')
print (classification_report(y_test, y_predicted_DT))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print (' \n confusion_matrix SGD \n ')
cm = confusion_matrix(y_test, y_predicted_SGD)
print (cm)
print ('\n Here is the classification report:')
print (classification_report(y_test, y_predicted_SGD))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print (' \n confusion_matrix SVM\n ')
cm = confusion_matrix(y_test, y_predicted_SVM)
print (cm)
print ('\n Here is the classification report:')
print (classification_report(y_test, y_predicted_SVM))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print (' \n confusion_matrix RFA \n ')
cm = confusion_matrix(y_test, y_predicted_RFA)
print (cm)
print ('\n Here is the classification report:')
print (classification_report(y_test, y_predicted_RFA))

