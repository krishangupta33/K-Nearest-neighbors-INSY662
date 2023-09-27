#KNN

#Task 2
#Create given dataset's dataframe using arrays
import pandas as pd

y=['black','blue','blue']
x1=[1,0,-1]
x2=[1,0,-1]

df=pd.DataFrame({'x1':x1,'x2':x2,'y':y})
print(df)

#Task 3
#K-NN model using k=2
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(df[['x1','x2']],df['y'])

#Task 4 
#Predicting the y value for x1=0.1 and x2=0.1
print(knn.predict([[0.1,0.1]]))


#Task 5
#Predicting using predict_proba method and finding probability of being black and blue
print(knn.predict_proba([[0.1,0.1]]))

#Task 6
#K-NN model with more parameters
knn=KNeighborsClassifier(n_neighbors=2,metric='euclidean')
knn=KNeighborsClassifier(n_neighbors=2,metric='manhattan')
knn=KNeighborsClassifier(n_neighbors=2,weights='distance')

