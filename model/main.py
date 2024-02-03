import pandas as pd
#importing satndard scaler to scale our data same scale
from sklearn.preprocessing import StandardScaler
#split the data
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,classification_report
 
import pickle as pick





#creatin a model 
def create_model(df):
    #dividing data into predictor and target
    x=df.drop(['diagnosis'],axis=1)#predictor
    y=df['diagnosis']


    #scale the data
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    
    #split data    
    x_train,x_test,y_train,y_test=train_test_split(
         x , y , test_size=0.2, random_state=42
    )

    #train
    model = LogisticRegression()
    model.fit(x_train,y_train)
    #test
    y_pred=model.predict(x_test)
    print('Accuracy of model:-',accuracy_score(y_test,y_pred))
    print('classification Report:-\n',classification_report(y_test,y_pred))

    return model ,scaler

def get_clean_data():
    df= pd.read_csv("data/data.csv")
    df= df.drop(['Unnamed: 32','id'],axis=1)
    df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

    print(df.head())
    return df
    
def main():
    df= get_clean_data()

    #training our model
    model,scaler= create_model(df)
    with open('model/model.pkl','wb') as f:
        pick.dump(model,f)
    with open('model/scaler.pkl','wb') as f:
        pick.dump(scaler,f)   
    #dumping peoject as file binary form
    #and these files we will impoer in our model









    
if __name__ == '__main__':
    main()    