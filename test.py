import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans


if __name__ == '__main__':
  
  X = np.zeros(shape=(200,4))

  Feature1_1 = np.random.normal(loc=40, scale=1.0, size=100)
  Feature1_2 = np.random.normal(loc=70, scale=3.0, size=100)

  Feature2_1 = np.random.normal(loc=20, scale=4.0, size=100)
  Feature2_2 = np.random.normal(loc=50, scale=1.0, size=100)

  Feature3_1 = np.random.normal(loc=40, scale=300.0, size=100)
  Feature3_2 = np.random.normal(loc=43, scale=280.0, size=100)

  Feature4_1 = np.random.normal(loc=20, scale=70.0, size=100)
  Feature4_2 = np.random.normal(loc=22, scale=70.0, size=100)


  X[:100,0]=Feature1_1
  X[100:,0]=Feature1_2
  X[:100,1]=Feature2_1
  X[100:,1]=Feature2_2
  X[:100,2]=Feature3_1
  X[100:,2]=Feature3_2
  X[:100,3]=Feature4_1
  X[100:,3]=Feature4_2
  
  #Created a dataset with 200 points and 4 features. The features 3 and 4 are created
  #with a larger variance hence are expected to be of higher importance during clustering
  
  
  kmeans = KMeans(n_clusters=2, n_init="auto")
  kmeans.fit(X)
  
  #Run core algo to get importance of each feature. 
  print(silhouette_feature_importance(X, kmeans.labels_))
  
  
