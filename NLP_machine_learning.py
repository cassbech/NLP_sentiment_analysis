# ! pip install unidecode
# ! pip install unicode
from NLPLibrary import *

#import labelled tweets excel file. depending on labelling, it can have 'ironie' & 'pub' columns besides from "tonalité"
from google.colab import files
uploaded=files.upload()

#_______read file, prepare dataset and create words matrix_______
def main():
    # data_tweet_sncf.xlsx
    fichier = input("Quel est le nom complet de votre fichier au format excel : Attention à ne pas mettre de guillemets, exemple : data_tweet_sncf.xlsx . Votre réponse : " )
    users = input('Si vous disposez d\'une colonne \"users\" en plus de vos colonnes \"tweets\" et \"tonalité\" tapez oui sinon tapez non : ')
    pub = input('Si vous disposez d\'une colonne \"pub/presse\" en plus de vos colonnes \"tweets\" et \"tonalité\" tapez oui sinon tapez non  : ')
    
    if (users == "oui"):
        if (pub=="oui"):
            data_frame = pd.read_excel(fichier,usecols=["users","tweets",'tonalité',"pub/presse"])
            data_frame["All_tweets"] = data_frame["tweets"].copy()
            data_frame.rename(index=str, columns={"users": "Users", "tweets": "Tweets traités","tonalité":"Tonalité","pub/presse":"Pub"},inplace=True)
            users = data_frame["Users"].tolist()
        else:
            data_frame = pd.read_excel(fichier,usecols=["users","tweets",'tonalité'])
            data_frame["All_tweets"] = data_frame["tweets"].copy()
            data_frame.rename(index=str, columns={"users": "Users", "tweets": "Tweets traités","tonalité":"Tonalité"},inplace=True)
            users = data_frame["Users"].tolist() 
    else:
        if (pub=="oui"):
            data_frame = pd.read_excel(fichier,usecols=["tweets","tonalité","pub/presse"])
            data_frame["All_tweets"] = data_frame["tweets"].tolist()
            data_frame.rename(index=str, columns={"tweets": "Tweets traités","tonalité":"Tonalité","pub/presse":"Pub"},inplace=True)
        else:
            data_frame = pd.read_excel(fichier,usecols=["tweets","tonalité"])
            data_frame["All_tweets"] = data_frame["tweets"].tolist()
            data_frame.rename(index=str, columns={"tweets": "Tweets traités","tonalité":"Tonalité"},inplace=True)

    data_frame["Tweets traités"] = data_frame["Tweets traités"].str.lower().astype(str)
    data_frame["Tweets traités"] = data_frame["Tweets traités"].apply(remove_url).apply(filter_tab).apply(remove_at).apply(remove_ponctuation).apply(remove_stop_words).apply(remplace_accents).apply(remove_ponctuation)
    
    list_of_words = word_occurency(data_frame)
    result = matrix_of_termes(data_frame,list_of_words)
    
    if (users == "non"):
        result["Users"] = users
    
    if(pub=="non"):
        result["Pub"]=0
        result["Pub"].loc[result[result["Tonalité"].isnull()].index.tolist()]=1
        result["Tonalité"]=result["Tonalité"].fillna("2")
        result["Tonalité"].replace(" ","2")
        cols = result.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        result = result[cols]  
    elif (pub =="oui"):
        result["Pub"] = result["Pub"].fillna("0")
    
    result["Tonalité"] = result["Tonalité"].astype(int)
    result["Pub"] = result["Pub"].astype(int)
    return result

resultat = main()

#_______machine learning_______
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score

X = resultat.drop(["Users", "Tweets traités", "Tonalité", "Pub", "All_tweets"], axis = 1)
y = resultat["Tonalité"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models_to_try={'Random Forest':RandomForestClassifier(n_estimators=100, max_features="auto", random_state=0),
              'Decision Tree':DecisionTreeClassifier(),
              'Bagging Classifier':BaggingClassifier(n_estimators=100),
              'Adaptive Boosting':AdaBoostClassifier(n_estimators=100),
              'Gradient Boosting':GradientBoostingClassifier(n_estimators=100),
              'Support Vector Machine':LinearSVC(),
              'Naive Bayes':naive_bayes.MultinomialNB()}

for name,algo in models_to_try.items():
  print(name)
  algo.fit(X_train, y_train)
  y_pred = algo.predict(X_test)
  print("Le score de précision est de {} %".format(round((accuracy_score(y_test, y_pred)*100), 2)))
  total_CV_scores= cross_val_score(algo, X, y, cv=10)
  CV_score=total_CV_scores.mean()
  print("Cross Validation Score: {} %".format(round(CV_score*100, 2)))
  print("{} tweets négatifs.".format(np.count_nonzero(y_pred == 1)))
  print("{} tweets neutres.".format(np.count_nonzero(y_pred == 2))) 
  print("{} tweets positifs.".format(np.count_nonzero(y_pred == 3)))
  
#_______scores summary_______
print("Decision Tree: {}%".format(round(dectree_score*100), 2))
print("Random Forest: {}%".format(round(forest_score*100), 2))
print("Bagging Classifier: {}%".format(round(bag_score*100), 2))
print("Adaptive Boosting: {}%".format(round(ada_score*100), 2))
print("Gradient Boosting: {}%".format(round(gbo_score*100), 2))
print("Support Vector Machine: {}%".format(round(svm_score*100), 2))
print("Naive Bayes: {}%".format(round(naive_score*100), 2))
