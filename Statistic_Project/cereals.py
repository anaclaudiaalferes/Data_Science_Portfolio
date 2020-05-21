import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples 
from sklearn.metrics import silhouette_score
import patsy
import statsmodels.api as sm
import pylab as py
import plotly.graph_objects as go
from plotly.offline import plot
cereal = pd.read_csv("https://raw.githubusercontent.com/pedromcsantos/Cereal_Stats/master/cereal.csv")


################ DATA EXPLORATION ################

cereal.isnull().sum() #no nulls but some -1
#if the -1 is in potassium its fine?
# if the -1 is in something caloric, its not fine
#oats are the only
cereal.drop(57, inplace = True)
cereal.reset_index(inplace = True, drop = True)
#Fill the -1 potassium with something. Avg from manufacturer? knn fit predict?


###Normalize Nutritional Facts with the weight
#do we take cups into the standardization?
#cereal A has weight = 1 and cup = 0.75, means that 1 serving is 0.75 of a cup and it weights 1 ounce.
#if every serving is standardized for 1 ounce, it means that the amount of that cup t get to 1 ounce also changes
## Vitamins shouldnt be there because vitamins are according to a 0/25/100 scale >> so they are actually categorical
#CUPS IS IMPORTANT -> the bigger the serving size, the more u eat > better perception?
cereal["weight"] = cereal["weight"]*28.35 #convert weight to grams
cereal[["sodium","potass"]] = cereal[["sodium","potass"]]/1000  #convert sodium, potassium to grams
#normalize weight to have comparable nutrients/calories
cereal.iloc[:,~cereal.columns.isin(['name','mfr','type',"shelf","vitamins",'rating'])] = cereal.iloc[:,~cereal.columns.isin(['name','mfr','type',"shelf","vitamins",'rating'])].div(cereal.weight, axis=0)
#Multiply by 100 to have readable number/measurement
cereal.iloc[:,~cereal.columns.isin(['name','mfr','type',"shelf","vitamins",'rating'])] = cereal.iloc[:,~cereal.columns.isin(['name','mfr','type',"shelf","vitamins",'rating'])]*100
#Drop weight because not necessary anymore
cereal.drop("weight", axis=1, inplace=True)

#Compute new calories which should be more accurate math wise and replace old?
for i in cereal.index:
    cereal.loc[i, "calories"] = 4*cereal.loc[i, "protein"] + 4*cereal.loc[i,"fiber"] + 4*cereal.loc[i,"sugars"] + 4*cereal.loc[i, "carbo"] + 9*cereal.loc[i, "fat"]


#Still have negative values for Potassium. Fill it with the mean for now.
cereal["potass"][cereal["potass"]<0] = cereal["potass"][cereal["potass"]>=0].mean()
 

#Drop type as it is not relevant -> Hot and Cold Cereals -> most of the cereals are cold 
cereal.drop("type", inplace = True, axis = 1)


cereal.drop(axis=0, index=[0,2,3], inplace=True) #eliminate these guys 

#Drop categorical variables for standardization: Name,  Manufacturer , vitamins, shelf

cereal_num = cereal.drop(["name","mfr","vitamins","shelf"], axis=1)


#Correlation Heatmap
corr = cereal_num.corr()
mask = np.zeros(corr.shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(corr, cmap = "Greens_r", annot = False, mask = mask,vmin = -1, vmax = 1,ax=ax,
            cbar_kws={'label': 'Correlation'})
#plt.savefig("Correlation.png")


#Standardize through ZSCORE

cereal.reset_index(drop=True, inplace=True)

scaler = StandardScaler()
cereals_scale = cereal_num.copy()
cereals_scale[cereals_scale.columns] = scaler.fit_transform(cereals_scale[cereals_scale.columns])



#Plot QQ PLOT Normalized vs QQ PLOT not Normalized
#https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
sm.qqplot(cereal["rating"],line="s") #which dataframe? One variable or all?
py.show() 

sm.qqplot(cereals_scale["potass"],line="s") #which dataframe? One variable or all?
py.show()


################ PCA ################

#we must first subtract the mean of each variable from the dataset to center the data around the origin. 
#Then, we compute the covariance matrix of the data and calculate the eigenvalues and corresponding 
#eigenvectors of this covariance matrix. Then we must normalize each of the orthogonal eigenvectors to 
#become unit vectors. Once this is done, each of the mutually orthogonal, unit eigenvectors can be 
#interpreted as an axis of the ellipsoid fitted to the data. 

#SKLEARN centers the data but doesnt standardize.
cereals_scale2 = cereals_scale.copy()
cereals_scale.drop("rating",axis=1, inplace=True) #drop target Y
pca = PCA(n_components= cereals_scale.shape[1])
principalComponents = pca.fit_transform(cereals_scale)
pca_cereals = pd.DataFrame(principalComponents)
# "Multivariate Analysis" by Hair et al (2012) 60percent Medical 95%
pca.explained_variance_ratio_
per_var = pca.explained_variance_ratio_

plt.plot(pca.explained_variance_ratio_)
plt.show()
for i in range(20):
    a=sum(pca.explained_variance_ratio_[:i])
    print(a) #8 principal components

#Scree Plot
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels, color = "green" )
plt.ylabel('Percentange of explained variance')
plt.xlabel('Principal component')
plt.title('Scree plot')
plt.show()
#plt.savefig("Screeplot.png")


loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings = pd.DataFrame(loadings, index= cereals_scale.columns.values)
loadings = loadings.round(3)


#transpose loadings for better reading
loadings_trans = loadings.transpose()
#If values above 0.5 means it has a strong correlation

#matrix where corr is at least 0.5 in 1 component

loadings_5 = loadings[(loadings >= 0.5) | (loadings <= -0.5)].dropna(axis=1, how = "all")
# these are the most important variables according to PCA and each PCA explains X amount of the variable

################ Clustering ################

#CLUSTER K MEANS using df cereals_scale which is normalized

cereals_clusters = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='random', random_state=1).fit(cereals_scale2)
    cereals_clusters.append(kmeans.inertia_)
    print(i) #check what iteration we are in


plt.plot(range(1,10), cereals_clusters)	
plt.show()

kmeans= KMeans(n_clusters=3, init='random', random_state=1).fit(cereals_scale2)
###3/6 clusters seem to have the best result.
# Inverse Normalization for Interpretation
cluster_centroids_num = pd.DataFrame(scaler.inverse_transform(X = kmeans.cluster_centers_))
cluster_centroids_num.columns = cereals_scale2.columns


cereal["cluster"] = kmeans.predict(X=cereals_scale2)
cereal.cluster.value_counts()

#COMPUTE THE SILHOUTTE

n_clusters=3
silhouette_avg = silhouette_score(cereals_scale2, kmeans.labels_)
print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(cereals_scale2, kmeans.labels_)

cluster_labels = kmeans.labels_

import matplotlib.cm as cm
y_lower = 100

fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i=ith_cluster_silhouette_values. shape[0]
    y_upper = y_lower + size_cluster_i
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples
    plt.show()



################ Linear Regression ################

#target Y = rating
 #remove calories as it is a linear combinar of  everything else

#y, x = patsy.dmatrices("rating~protein+fat+sodium+fiber+carbo+sugars+potass+C(vitamins)+C(shelf)+cups+C(mfr)+C(cluster)",cereal)
y, x = patsy.dmatrices("rating~protein+fat+sodium+fiber+carbo+sugars+potass",cereal) 

pca_regression = pca_cereals.copy()

pca_regression.rename({0:"PC1",1:"PC2",2:"PC3", 3:"PC4"},axis=1, inplace = True)

pca_regression["rating"] = cereal["rating"]

y, x = patsy.dmatrices("rating~protein+fat+sodium+fiber+carbo+sugars+potass",cereal) 



model= sm.OLS(y,x).fit()
print(model.summary())  #good model with good fit where fiber seems to increase the rating the most while fat decreases, cp.

y, x = patsy.dmatrices("rating~PC1+PC2",pca_regression) 
model= sm.OLS(y,x).fit()
print(model.summary()) 

y, x = patsy.dmatrices("rating~PC2",pca_regression) 
model= sm.OLS(y,x).fit()
print(model.summary()) 

    
plt.scatter(x=cereal.sugars, y = cereal.rating) #the more the sugar, the lower the rating
plt.scatter(x=cereal.fat, y = cereal.rating)  #the more the sugar, the lower the rating
plt.scatter(x= cereal.cups , y= cereal.calories)
plt.scatter(x= cereal.fiber , y= cereal.rating)
plt.scatter(x= cereal.protein , y= cereal.rating)

plt.scatter(x=pca_regression.iloc[:,1], y = pca_regression.rating) 
   
 
plt.hist(cereal["rating"]) #somewhat right skewed distribution
plt.axvline(cereal.rating.mean(), c = "black")
plt.show()

c= pca_regression.PC1
d=pca_regression.PC2
sns.regplot(x="PC2", y="rating", data=pca_regression, color="green")
