# -*- coding: utf-8 -*-

"""
Created on Tue Nov 19 14:53:23 2019
@author: GroupS
"""

###############
'''IMPORTS'''
###############


import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

import joblib
import random

from sompy.sompy import SOMFactory
from sompy.visualization.plot_tools import plot_hex_map
import logging

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz 

from kmodes.kmodes import KModes
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Notice when you just run it doesn't show all the plots,
# we just make visible the ones we consider most important for our final decision

###############
'''FUNCTIONS'''
###############

def elbow(data, seeds_to_test):
    """
    Function that plots elbow graph
    Params: data (DataFrame) with the dataset
            seeds_to_test (Int) corresponding to the number of seeds that we want to feed
    """
    wcss = []
    # calculating error for different number of seeds for clusteringaround 60% are explained by first 3 PCA's, then each additional PCA explains another 10%
    for i in range(1,seeds_to_test+1):
        kmeans = KMeans(n_clusters = i, random_state = 0)
        kmeans.fit(data)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
        # plotting the number of clusters vs error and choosing the number of clusters based on the elbow method
    number_cluster = range(1, seeds_to_test+1)
    plt.plot(number_cluster, wcss)


def siltest(df, labels, n_clu, clusters, dfOriginal):
    """
    Function that plots silhouette graph and calculets the mean distance 
    between points and centroids
    Params: df(DataFrame) 
            lables(Numpy Array) labels of clusters 
            n_clu(Int) correponding the number of clusters
            clusters(DataFrame) corresponding to the dataframe with the centroids
            dfOriginal(DataFrame) corresponding to the original dataframe with the points
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    silhouette_avg = silhouette_score(df, labels)
    
    print("For n_clusters =", n_clu,
              "The average silhouette_score is :", silhouette_avg)
    
    from scipy.spatial.distance import cdist
    dm = cdist(dfOriginal, clusters.drop(['Clients'], axis=1))
    print("For n_clusters =", n_clu,
          "The Distance Mean:", dm.mean())
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, labels)
    cluster_labels = labels
    
    import matplotlib.cm as cm
    y_lower = 100
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, policies_scaled.shape[0] + (n_clu + 1) * 10])
    
    for i in range(n_clu):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.rainbow(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color, 
                          alpha=0.7)
    
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 100
        
        ax.set_title("The silhouette analysis with various n clusters")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        
    plt.show()


def dens_scan(epsilon, samples, data):
    db = DBSCAN(eps=epsilon,
                min_samples=samples).fit(data)

    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_clusters, counts_clusters = np.unique(db.labels_, return_counts=True)
    print(np.asarray((unique_clusters, counts_clusters)))
    return labels


def dens_scan_iter(start_epsilon, samples, data, step_e, step_s, iterations_e, iterations_s):
    lists = []
    for i in range(1, iterations_e):
        for j in range(1, iterations_s):
            lists.append(dens_scan(start_epsilon + step_e * i, samples * step_s * j, data))
    return lists

def retrieving_info_noise(clusters_iter):
    noise = []
    for cluster in clusters_iter:
        counts = np.count_nonzero(cluster == -1)
        noise.append(counts)
    return noise

    
    
###########################################  
        
my_path= 'A2Z Insurance.csv'
mydata = pd.read_csv(my_path)

###########################################
'''PREPROCESSING'''
###########################################

mydata.info()
mydata.isnull().sum()#Checking completeness of dataset
#All our variables have enough values and can be used for input area for our segmentation \

'''
The nan values are less than 0.5% out of 10K clients
so all variables are fit to be used for clustering (data is complete).
'''
#renaming for simplicity
mydata.rename(columns={"Customer Identity": "ID", "First Policy´s Year": "1stPol", "Educational Degree": "Edu","Gross Monthly Salary": "Salary","Geographic Living Area": "Area",
                       "Has Children (Y=1)": "Children","Customer Monetary Value": "CMV","Claims Rate": "Claims Rate",
                       "Premiums in LOB: Motor": "Car", "Premiums in LOB: Household": "Home",
                       "Premiums in LOB: Health": "Health","Premiums in LOB:  Life": "Life",
                       "Premiums in LOB: Work Compensations": "Work"}, inplace=True)


mydata['YearsofPol']= 2016 - mydata['1stPol']
mydata['Edu']= mydata['Edu'].str.strip()
mydata['Education ID']= mydata['Edu'].str[:2]
mydata['Education ID']= pd.to_numeric(mydata['Education ID'])
mydata['TotalPremium']= mydata.iloc[:,9:14].sum(axis=1)  #we only need this calculated column to filter out outliers, we do not use it for clustering


#adding policies count as we believe it might be a useful variable for clustering
#if there is negative payment it is not taken into count as client stopped the policy
mydata['Count']= mydata[mydata.iloc[:,9:14].replace({0:np.nan})>0].count(axis=1)

#we replaced nan values with zeros for policy payments since we consider that:
#Absence of payment = zero payment
mydata['Car']= mydata['Car'].fillna(0)
mydata['Home']= mydata['Home'].fillna(0)
mydata['Health']= mydata['Health'].fillna(0)
mydata['Life'] = mydata['Life'].fillna(0)
mydata['Work']= mydata['Work'].fillna(0)
mydata.groupby('Count').count()['ID']

#mydata['Age']= 2016- mydata['Brithday Year'] 
mydata= mydata.drop(['Brithday Year'], axis=1) #age column does not make sense together with first policy year
mydata.info()

#this is original df without nulls and without categorical values for DBSCAN
nulls=mydata[mydata.isna().any(axis=1)] # null values 78 clients
nonulls = mydata.dropna()
nonulls.drop(['Edu'],axis=1,inplace=True)
nonulls.drop(['Children'],axis=1,inplace=True)
nonulls.drop(['Area'],axis=1,inplace=True)
nonulls.drop(['Education ID'],axis=1,inplace=True)
nonulls_scaled = pd.DataFrame(scaler.fit_transform(nonulls))
nonulls_scaled.columns = nonulls.columns

########
'''CORRELATION 1'''
#########
#performed on full dataset stripped of null values
corr = mydata.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

###########################################
'''DATA EXPLORATION'''

#basic stats
statsnonulls=nonulls.describe()
statsnonullsst=nonulls_scaled.describe()

#explore features by categories existing in our clients set
#who are clients by area?
areainsights=mydata.dropna().groupby('Area').mean()
mydata.dropna().groupby(["Area", "Children"])["ID"].count()
mydata.dropna().groupby(["Edu","Area"])["ID"].count()
mydata.dropna().groupby(["Area"])["ID"].count()

test=mydata.dropna().groupby("Area")['Car','Home','Health','Life','Work'].agg(["mean", "median"])
test=mydata.dropna().groupby("Area")
#who are clients by children?
kidsinsights=mydata.dropna().groupby('Children').mean()
#who are clients by education level?
eduinsights=mydata.dropna().groupby("Edu").mean()
mydata.dropna().groupby("Edu")["ID"].count() 

###########################
'''OUTLIERS'''
##########################

#We tested the quartiles method and decided it is not suitable for our dataset as it cuts off too many outliers: 
#quartiletest1 = mydata['CMV'] # this returned 206 outliers
#removed_outliers = quartiletest1.between(quartiletest1.quantile(.01), quartiletest1.quantile(.99))

#quartiletest2 = mydata['Claims Rate']  # this returned 189 outliers
#removed_outliers = quartiletest2.between(quartiletest2.quantile(.01), quartiletest2.quantile(.99))

#The next method tested was boxplot
#boxplot = mydata.boxplot(column=['Salary']) #any value over 10000 is outlier
outliers= mydata[mydata['Salary'] > 30000]

#boxplot = mydata.boxplot(column=['CMV']) #any value over 10000 or below -25000 is outlier 
outliers= outliers.append(mydata[(mydata['CMV'] < -25000 ) | (mydata['CMV'] > 10000)])

#boxplot = mydata.boxplot(column=['Claims Rate'],whis=[5, 95]) #any value over 20 is outlier 
outliers= outliers.append(mydata[mydata['Claims Rate'] > 20])

#boxplot = mydata.boxplot(column=['TotalPremium']) #any value over 20000 is outlier 
outliers= outliers.append(mydata[mydata['TotalPremium'] > 20000])

#boxplot = mydata.boxplot(column=['YearsofPol']) #any value below 0 is outlier 
outliers= outliers.append(mydata[mydata['YearsofPol'] < 0 ])

#boxplot = mydata.boxplot(column=['Car'])
outliers= outliers.append(mydata[mydata['Car'] >2000])
outliers= outliers.append(mydata[mydata['Home'] >8000])
outliers= outliers.append(mydata[mydata['Health'] >7000])
outliers= outliers.append(mydata[mydata['Work'] >1000])

#adding null lines to Outliers dataframe
outliers= outliers.append(mydata[(mydata['1stPol'].isnull()) | (mydata['Edu'].isnull())
| (mydata['Salary'].isnull())| (mydata['Area'].isnull())
| (mydata['Children'].isnull())| (mydata['CMV'].isnull())
| (mydata['YearsofPol'].isnull())| (mydata['Education ID'].isnull())])

#final outliers list with unique clients ID
outliers.drop_duplicates(subset='ID',inplace=True)

#dropping outliers and nulls to get a clean df, 10197 clients
clean= mydata[mydata['ID'].isin(outliers['ID']) == False]
   
clean.isnull().sum() #Check for null values
clean.reset_index(drop=True,inplace=True)
outliers.reset_index(drop=True,inplace=True)


########
'''CORRELATION 2'''
#########
#Check correlation without outliers
#corr = clean.corr()
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#f, ax = plt.subplots(figsize=(11, 9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

#Check correlation on subset of policies
policies =clean.loc[:,['Car','Home','Health','Life','Work']]
mask = np.zeros_like(policies.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#plt.figure(figsize=(15,15))
#sns.heatmap(policies.corr(),annot=True,mask=mask,cmap="YlGnBu",vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#plt.show()

#Check correlation on subset customer numerical
custnum=clean.loc[:,['Salary','Claims Rate','YearsofPol','CMV','Count']]
mask = np.zeros_like(custnum.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#plt.figure(figsize=(15,15))
#sns.heatmap(custnum.corr(), annot=True,mask=mask,cmap="YlGnBu",vmax=.3, center=0,
#            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#plt.show()

#Get basic stats on clean df
cleanstats=clean.describe()

##########
'''STANDARDIZE DATA
The standardization is applied after the removal of outliers and after the treatment of missing values.
'''
##########
# We divide our features into 3 groups
# 1 Policies
scaler_pol = StandardScaler()
policies =clean.loc[:,['Car','Home','Health','Life','Work']] # consider adding TotalPremium
policies_scaled = pd.DataFrame(scaler_pol.fit_transform(policies))
policies_scaled.columns = policies.columns

# 2 Customer numeric
scaler_cust = StandardScaler()
custnum=clean.loc[:,['Salary','CMV','YearsofPol','Count']] # added count of policies and removed CMV as highly correlated to Claims rate
custnum_scaled = pd.DataFrame(scaler_cust.fit_transform(custnum))
custnum_scaled.columns = custnum.columns

# 3 Customer categorical
custcat=clean[['Area','Children','Education ID']].astype('str')

###############
'''PCA ANALYSIS on numerical variables'''
#############
#Check for the variable with highest explanatory power
my_data_num=policies_scaled.merge(custnum_scaled,right_index=True, left_index=True)
pca = PCA(n_components= my_data_num.shape[1]) # pca count = count of variables(10)
principalComponents = pca.fit_transform(my_data_num)
                                        
# Plot the explained variances
features=range(pca.n_components_)
#plt.bar(features, pca.explained_variance_ratio_, color='green')
#plt.xlabel('PCA features')
#plt.ylabel('variance %')
#plt.xticks(features)
#plt.title('PC Analysis of numerical variables')

pca.components_ # list of eigen values for each PCA
pca.singular_values_
pca.inverse_transform(principalComponents)

#heatplot to see how the features mixed up to create the components
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2,3,4,5,6,7,8],['1st Comp','2nd Comp','3rd Comp', '4th Comp','5th Comp',
           '6th Comp','7th Comp','8th Comp','9th Comp'],fontsize=8)
plt.colorbar()
plt.xticks(range(len(my_data_num.columns)),my_data_num.columns,rotation=65,ha='left')
plt.tight_layout()
plt.show()

#PCA on only policies
pca=PCA(n_components= policies_scaled.shape[1])
principalComponents = pca.fit_transform(policies_scaled)
features=range(pca.n_components_)
#plt.bar(features, pca.explained_variance_ratio_, color='green')
#plt.xlabel('PCA features')
#plt.ylabel('variance %')
#plt.title('PC Analysis of policies')
#plt.xticks(features)

#heatplot to show dependencies on original features
#plt.matshow(pca.components_,cmap='viridis')
#plt.yticks([0,1,2,3,4],['1st Comp','2nd Comp','3rd Comp', '4th Comp','5th Comp'],fontsize=8)
#plt.colorbar()
#plt.xticks(range(len(policies_scaled.columns)),policies_scaled.columns,rotation=65,ha='left')
#plt.title('PCs and dependencies on original features')
#plt.tight_layout()
#plt.show()

#PCA on customer info
pca=PCA(n_components= custnum_scaled.shape[1])
principalComponents = pca.fit_transform(custnum_scaled)
features=range(pca.n_components_)
#plt.bar(features, pca.explained_variance_ratio_, color='green')
#plt.xlabel('PCA features')
#plt.ylabel('variance %')
#plt.title('PC Analysis of customer numerical features')
#plt.xticks(features)

#heatplot to show dependencies on original features
#plt.matshow(pca.components_,cmap='viridis')
#plt.yticks([0,1,2,3],['1st Comp','2nd Comp','3rd Comp', '4th Comp'],fontsize=8)
#plt.colorbar()
#plt.xticks(range(len(custnum_scaled.columns)),custnum_scaled.columns,rotation=65,ha='left')
#plt.title('PCs and dependencies on original features')
#plt.tight_layout()
#plt.show()


##########################################
'''CLUSTERING'''
##########################################

###############################################
'''DBSCAN'''
###############################################

#Now we are running DBSCAN with several sets of epsilons to re-confirm outliers
clusters_iter = dens_scan_iter(1, 30, nonulls_scaled, 1, 10, 7, 5)
noise = retrieving_info_noise(clusters_iter)

objects = np.unique(noise)
y_pos = np.arange(len(objects))
_ , count = np.unique(noise, return_counts = True)
#plt.bar(y_pos,count, align='center',alpha = 0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('# of times same noise value')
#plt.title('DBSCAN noise (outliers)')
#plt.show()

   
###############################
"""POLICIES KMEANS CLUSTERS"""
##############################

#elbow grapg on scaled policies:
#elbow(policies_scaled, 10)

#After running the elbow function the number of clusters to decide on is between 2,3 or 4
#Once we tried with 4,3 and 2 clusters we decided to stick with only 2 clusters for K-Means aproach

n_clusters = 2

kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0,
                n_init = 50,
                max_iter = 500).fit(policies_scaled)

policies_scaled_clusters = pd.DataFrame(kmeans.cluster_centers_)

# Check the Clusters
#reverse to original de-standardized values
policies_scaled_clusters = pd.DataFrame(scaler_pol.inverse_transform(X = policies_scaled_clusters),
                                      columns= policies_scaled.columns)

#labeling columns and adding kmeans cluster number to each client
policies_kmeans=pd.DataFrame(pd.concat([policies_scaled, pd.DataFrame(kmeans.labels_)],axis=1))
policies_kmeans.columns = ['Car','Home','Health','Life','Work','PolLabels']
policies_kmeans=policies_kmeans.dropna()
policies_scaled_clusters['Clients']=policies_kmeans.groupby('PolLabels').count()['Car'] # adding clients count per cluster

#silhouette and distance test
siltest(policies_scaled, kmeans.labels_ , n_clusters, policies_scaled_clusters ,policies)


#############################################
"""POLICIES Hierarchical CLUSTERS """
############################################# 
#Generate dendrograms to defined an appropriate number of clusters

#plt.figure(figsize=(10,5))
#plt.style.use('seaborn-whitegrid')
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#Z = linkage(policies_scaled,
#            method ='ward')#method='single', 'complete', 'ward'

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k'])
#dendrogram(Z,
#           truncate_mode='lastp',
#           p=10,
#           orientation = 'top',
#           leaf_rotation= 45.,
#           leaf_font_size= 10.,
#           show_contracted= True,
#           show_leaf_counts=True, color_threshold=50, above_threshold_color='k')


#plt.title('Truncated Hierarchical Clustering Dendrogram')
#plt.xlabel('Cluster Size')
#plt.ylabel('Distance')
#plt.axhline(y=50)
#plt.show()

#After running the dendogram the number of clusters to decide on is between 2,3 or 4
#We decided to stick with 4 clusters for Hierarchical aproach

k = 4

Hclustering = AgglomerativeClustering(n_clusters= k,
                                      affinity= 'euclidean',
                                      linkage= 'ward')

#Replace the test with proper data
policies_scaled_HC = Hclustering.fit(policies_scaled)

my_labels_policies_HC = pd.DataFrame(policies_scaled_HC.labels_)
my_labels_policies_HC.columns =  ['Labels']

policies_HC = pd.DataFrame(pd.concat([pd.DataFrame(policies_scaled),my_labels_policies_HC],axis=1), 
                        columns=['Car','Home','Health','Life','Work','Labels'])


# Do the necessary transformations
final_result_policies_scaled_HC = policies_HC.groupby('Labels').mean()

my_clusters_policies_HC = pd.DataFrame(scaler_pol.inverse_transform(X = final_result_policies_scaled_HC), columns = final_result_policies_scaled_HC.columns )
my_clusters_policies_HC['Clients'] = my_clusters_policies_HC['Car']+my_clusters_policies_HC['Home']+my_clusters_policies_HC['Health']+my_clusters_policies_HC['Life']+my_clusters_policies_HC['Work']
policies_HC.groupby('Labels').count()

#silhouette and distance test
#siltest(policies_scaled, policies_scaled_HC.labels_, k ,my_clusters_policies_HC ,policies)


#############################################
"""POLICIES SOM CLUSTERS """
#############################################

X = policies_scaled.values

names = ['Car','Home','Health','Life','Work']

sm = SOMFactory().build(data = X,
               mapsize=(10,10),
               normalization = 'var',
               initialization='random',
               component_names=names,
               lattice='hexa',
               training = 'seq')

sm.train(n_job=4,
         verbose='info',
         train_rough_len=30, # the first half of distance is covered by bigger steps
         train_finetune_len=100) # the remaining half of diatance covered by smaller steps = finetuning

final_clusters = pd.DataFrame(sm._data, columns = ['Car','Home','Health','Life','Work'])

my_labels = pd.DataFrame(sm._bmu[0])
    
final_clusters = pd.concat([final_clusters,my_labels], axis = 1)
final_clusters.columns = ['Car','Home','Health','Life','Work', 'Labels']

from sompy.visualization.mapview import View2DPacked
#view2D  = View2DPacked(10,10,"", text_size=7)
#view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
#plt.show()

from sompy.visualization.mapview import View2D
#view2D  = View2D(10,10,"", text_size=7)
#view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
#plt.show()

from sompy.visualization.bmuhits import BmuHitsView
#vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
#vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="autumn", logaritmic=False)

from sompy.visualization.hitmap import HitMapView
#sm.cluster(4)
#hits  = HitMapView(10,10,"Clustering",text_size=7)
#a=hits.show(sm, labelsize=12)

##############################
'''APPLYING DENDOGRAM OVER SOM '''
###############################

#Calculation of centroids features
som_neurons= final_clusters.groupby('Labels').mean()

#Denormalize centroids values
som_neurons= pd.DataFrame(scaler_pol.inverse_transform(som_neurons),columns=som_neurons.columns)

#Generate dendrograms to defined an appropriate number of clusters

#Z = linkage(som_neurons,
#            method ='ward')#method='single', 'complete', 'ward'

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k'])
#dendrogram(Z,
#           truncate_mode='lastp',
#           p=10,
#           orientation = 'top',
#           leaf_rotation= 45.,
#           leaf_font_size= 10.,
#           show_contracted= True,
#           show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

#plt.title('Truncated Hierarchical Clustering Dendrogram')
#plt.xlabel('Cluster Size')
#plt.ylabel('Distance')
#plt.axhline(y=50)
#plt.show()

#After running the dendogram the number of clusters to decide on is between 2,3 or 4
#Once we tried with 4,3 and 2 clusters we decided to stick with 3 clusters 
#for an Hierarchical over SOM aproach

k = 3

Hclustering = AgglomerativeClustering(n_clusters= k,
                                      affinity= 'euclidean',
                                      linkage= 'ward')

#Replace the test with proper data, create column with label produced by HC algorithm
HCmodel = Hclustering.fit(som_neurons) # function feeds avg clusters from SOM through dendogram
HCmodel_labels = pd.DataFrame(HCmodel.labels_) # produces list of lables which are 4 clusters we obtaned by HC

#table clusters_avg with new HC label column
SOMHCclusters = pd.DataFrame(pd.concat([pd.DataFrame(som_neurons),HCmodel_labels],axis=1))
SOMHCclusters.columns = ['Car','Home','Health','Life','Work', 'HC_Labels']
#final CLUSTERS avergare properties in original scale - check if clusters make sense
SOMHCclusters_avg = SOMHCclusters.groupby('HC_Labels').mean()
SOMHCclusters_avg['Clients'] = SOMHCclusters_avg['Car']+SOMHCclusters_avg['Home']+SOMHCclusters_avg['Health']+SOMHCclusters_avg['Life']+SOMHCclusters_avg['Work']

resultSOM = pd.merge(final_clusters, SOMHCclusters['HC_Labels'], left_on='Labels', right_index=True,how='left', sort=False)
resultSOM.groupby('HC_Labels').count()

#On this particular algorithm we decided to just plot the distance without the function "siltest"
from scipy.spatial.distance import cdist
dm = cdist(policies, SOMHCclusters_avg.drop(['Clients'],axis=1))
dm.mean()

##########################
'''CUSTOMERS NUMERIC KMEANS'''
###########################

#elbow graph to decided the number of clusters for customers numeric
#elbow(custnum_scaled, 10)

#After running the elbow function the number of clusters to decide on is between 2,3 or 4
#Once we tried with 4,3 and 2 clusters we decided to stick with 3 clusters for customer information

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0,
                n_init = 50,
                max_iter = 2000).fit(custnum_scaled)

custnumclusters = pd.DataFrame(kmeans.cluster_centers_)

# Check the Clusters
custnumclusters = pd.DataFrame(scaler_cust.inverse_transform(custnumclusters),
                                      columns= custnum_scaled.columns)


cust_kmeans=pd.DataFrame(pd.concat([custnum, pd.DataFrame(kmeans.labels_)],axis=1))
cust_kmeans.columns = ['Salary','CMV','YearsofPol', 'Count','CustNumLabels']
cust_kmeans.groupby('CustNumLabels').count()
custnumclusters['Clients']=cust_kmeans.groupby('CustNumLabels').count()['Salary'] # adding clients count per cluster

#silhouette and distance test
siltest(custnum_scaled, kmeans.labels_,n_clusters,custnumclusters,custnum)

##########################
'''CUSTOMERS NUMERIC HIERARCHICAL'''
###########################
#Generate dendrograms to defined an appropriate number of clusters

#plt.figure(figsize=(10,5))
#plt.style.use('seaborn-whitegrid')

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#Z = linkage(custnum_scaled,
#            method ='ward')#method='single', 'complete', 'ward'

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k'])
#dendrogram(Z,
#           truncate_mode='lastp',
#           p=10,
#           orientation = 'top',
#           leaf_rotation= 45.,
#           leaf_font_size= 10.,
#           show_contracted= True,
#           show_leaf_counts=True, color_threshold=50, above_threshold_color='k')


#plt.title('Truncated Hierarchical Clustering Dendrogram')
#plt.xlabel('Cluster Size')
#plt.ylabel('Distance')
#plt.axhline(y=50)
#plt.show()

#After running the dendogram the number of clusters to decide on is between 2,3 or 4
#Once we tried with 4,3 and 2 clusters we decided to stick with only 4 clusters for an Hierarchical aproach

k = 4

Hclustering = AgglomerativeClustering(n_clusters= k,
                                      affinity= 'euclidean',
                                      linkage= 'ward')

#Replace the test with proper data
df_Cos_scaled_HC = Hclustering.fit(custnum_scaled)
my_labels_Cos_HC = pd.DataFrame(df_Cos_scaled_HC.labels_)
my_labels_Cos_HC.columns =  ['Labels']

Cos_HC = pd.DataFrame(pd.concat([pd.DataFrame(custnum_scaled),my_labels_Cos_HC],axis=1), 
                        columns=['Salary','CMV','YearsofPol','Count','Labels'])


# Do the necessary transformations
final_result_Cos_scaled_HC = Cos_HC.groupby('Labels').mean()
my_clusters_Cos_HC = pd.DataFrame(scaler_cust.inverse_transform(X = final_result_Cos_scaled_HC), columns = final_result_Cos_scaled_HC.columns )
my_clusters_Cos_HC ['Clients'] = Cos_HC.groupby('Labels').count().Count

Cos_HC.groupby('Labels').count()

#silhouette and distance test
#siltest(custnum_scaled, df_Cos_scaled_HC.labels_, k ,my_clusters_Cos_HC,custnum)

##########################
'''CUSTOMERS NUMERIC SOM'''
###########################
X = custnum_scaled.values

names = ['Salary','CMV','YearsofPol','Count']

sm = SOMFactory().build(data = X,
               mapsize=(10,10),
               normalization = 'var',
               initialization='random',
               component_names=names,
               lattice='hexa',
               training = 'seq')

sm.train(n_job=4,
         verbose='info',
         train_rough_len=30, # the first half of distance is covered by bigger steps
         train_finetune_len=100) # the remaining half of diatance covered by smaller steps = finetuning

final_clustersCUM = pd.DataFrame(sm._data, columns = ['Salary','CMV','YearsofPol','Count'])

my_labelsCUM = pd.DataFrame(sm._bmu[0])
    
final_clustersCUM = pd.concat([final_clustersCUM,my_labelsCUM], axis = 1)
final_clustersCUM.columns =['Salary','CMV','YearsofPol','Count','Labels']

from sompy.visualization.mapview import View2DPacked
#view2D  = View2DPacked(10,10,"", text_size=7)
#view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
#plt.show()

from sompy.visualization.mapview import View2D
#view2D  = View2D(10,10,"", text_size=7)
#view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
#plt.show()

from sompy.visualization.bmuhits import BmuHitsView
#vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
#vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="autumn", logaritmic=False)

from sompy.visualization.hitmap import HitMapView
#sm.cluster(4)
#hits  = HitMapView(10,10,"Clustering",text_size=7)
#a=hits.show(sm, labelsize=12)

##############################
'''APPLYING DENDOGRAM OVER SOM '''
###############################

#Calculation of centroids features (this is the input for HC)
som_neuronsCUM = final_clustersCUM.groupby('Labels').mean()

#Denormalize centroids values
som_neuronsCUM= pd.DataFrame(scaler_cust.inverse_transform(som_neuronsCUM),columns=som_neuronsCUM.columns)

#Generate dendrograms to defined an appropriate number of clusters

#Z = linkage(som_neurons,
#            method ='ward')#method='single', 'complete', 'ward'

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#hierarchy.set_link_color_palette(['c', 'm', 'y', 'g','b','r','k'])
#dendrogram(Z,
#          truncate_mode='lastp',
#          p=10,
#          orientation = 'top',
#          leaf_rotation= 45.,
#          leaf_font_size= 10.,
#          show_contracted= True,
#          show_leaf_counts=True, color_threshold=50, above_threshold_color='k')

#plt.title('Truncated Hierarchical Clustering Dendrogram')
#plt.xlabel('Cluster Size')
#plt.ylabel('Distance')
#plt.axhline(y=50)
#plt.show()

#After running the dendogram the number of clusters to decide on is between 2,3 or 4
#Once we tried with 4,3 and 2 clusters we decided to stick with 4 clusters 
#for an Hierarchical over SOM aproach


k = 4

Hclustering = AgglomerativeClustering(n_clusters= k,
                                      affinity= 'euclidean',
                                      linkage= 'ward')

#Replace the test with proper data, create column with label produced by HC algorithm
HCmodelCUM = Hclustering.fit(som_neuronsCUM) # function feeds avg clusters from SOM through dendogram
HCmodelCUM_labels = pd.DataFrame(HCmodelCUM.labels_) # produces list of lables which are 4 clusters we obtaned by HC

#table clusters_avg with new HC label column
SOMHCclustersCUM = pd.DataFrame(pd.concat([pd.DataFrame(som_neuronsCUM),HCmodelCUM_labels],axis=1))
SOMHCclustersCUM.columns = ['Salary','CMV','YearsofPol','Count','HC_Labels']

#final CLUSTERS avergare properties in original scale
SOMHCclustersCUM_avg = SOMHCclustersCUM.groupby('HC_Labels').mean()

resultSOMC = pd.merge(final_clustersCUM, SOMHCclustersCUM['HC_Labels'], left_on='Labels', right_index=True,how='left', sort=False)
resultSOMC.groupby('HC_Labels').count()

#On this particular algorithm we decided to just plot the distance without the function "siltest"
from scipy.spatial.distance import cdist
dm = cdist(custnum, SOMHCclustersCUM_avg)
dm.mean()

##########################
'''CUSTOMERS CATEGORICAL KMODES'''
###########################

km = KModes(n_clusters=2, init='random', n_init=50, verbose=1)

clusters = km.fit_predict(custcat)

#Print the kmodes cluster centroids:
#print(km.cluster_centroids_)
cat_centroids = pd.DataFrame(km.cluster_centroids_,columns = ['Area','Children','Education ID'])
unique, counts = np.unique(km.labels_, return_counts=True)
cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['CustCatLabel','Number'])
cat_centroids = pd.concat([cat_centroids, cat_counts], axis = 1)

#Add kmodes label to original customer list
cust_kmodes=pd.DataFrame(pd.concat([custcat, pd.DataFrame(km.labels_)],axis=1))
cust_kmodes.columns = ['Area','Children','Education ID','CustCatLabels']

#######################################
'''MERGING CLUSTER RESULTS and CONTINGENCY TABLE'''
#####################################
#Merging clustering results
clust_merged = clean[['ID']] # creating customer index as a separate column
clust_merged= clust_merged.merge(policies_kmeans[['PolLabels']],right_index=True,left_index=True)
clust_merged= clust_merged.merge(cust_kmodes[['CustCatLabels']],right_index=True,left_index=True)
clust_merged=clust_merged.merge(cust_kmeans[['CustNumLabels']],right_index=True,left_index=True)

#Creating list of clients with redflag
redflag = mydata[mydata['Health']<0]
redflag=redflag.append(mydata[mydata['Work']<0])
redflag=redflag.append(mydata[mydata['Life']<0])

#Combine customer value numeric and policies
#No categorical cluster used in this step
Crosstab = pd.crosstab(clust_merged['PolLabels'],clust_merged['CustNumLabels'])
Crosstab.columns = ['High', 'Low', 'Avg'] 
Crosstab.index = ["Other", "Car"]

#Combine results and describe
clust_merged_full=clust_merged.merge(clean[['Salary','CMV','Count','Claims Rate','Children','Work','Car','Home','Area','Edu', 'Life', 'Health','YearsofPol','TotalPremium']],right_index=True,left_index=True)

#Checking for insigths combining the categorical information with the policies and demographics  
#pivot=pd.crosstab(clust_merged_full['PolLabels'],clust_merged_full['Children'])#do they have kids?
#pivot.index = ['Other', 'Car'] #naming clusters
#pivot.columns = ['No kids', 'Kids'] #naming categories
#pivot=pd.crosstab(clust_merged_full['PolLabels'],clust_merged_full['Area'])
#clust_merged_full.groupby('CustNumLabels').mean()['TotalPremium']
#clust_merged_full.groupby('Edu').count()['PolLabels']
#clust_merged_full.groupby('Children').count()['PolLabels']

#Creating labels for our final 6 clusters presented in the variable "Crosstab"
clust_merged_full.loc[(clust_merged_full['PolLabels'] == 0) & (clust_merged_full['CustNumLabels'] == 0), 'FinalCluster'] = 0
clust_merged_full.loc[(clust_merged_full['PolLabels'] == 0) & (clust_merged_full['CustNumLabels'] == 1), 'FinalCluster'] = 1
clust_merged_full.loc[(clust_merged_full['PolLabels'] == 0) & (clust_merged_full['CustNumLabels'] == 2), 'FinalCluster'] = 2

clust_merged_full.loc[(clust_merged_full['PolLabels'] == 1) & (clust_merged_full['CustNumLabels'] == 0), 'FinalCluster'] = 10
clust_merged_full.loc[(clust_merged_full['PolLabels'] == 1) & (clust_merged_full['CustNumLabels'] == 1), 'FinalCluster'] = 11
clust_merged_full.loc[(clust_merged_full['PolLabels'] == 1) & (clust_merged_full['CustNumLabels'] == 2), 'FinalCluster'] = 12

clust_merged_full=clust_merged_full.drop(['CustCatLabels'],axis=1)

#########################################
"""REINTRODUCING OUTLIERS"""
########################################
#The point of building a model, is to classify new data with undefined labels. 
#Therefore, we will try to reassign the customer that we classified as outliers

X = clust_merged_full[['Salary','CMV','Count','Claims Rate','Children','Work','Car','Home','Area', 'Life', 'Health','YearsofPol','TotalPremium']] 
y =  clust_merged_full[['FinalCluster']] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1) # 70% training and 30% tes

######################################################################################## Knearest neighbours
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

#Using our newly trained model, we make a predicton.
y_pred = knn.predict(X_test)

"""A way of evaluating our model is to compute the confusion matrix. 
The numbers on the diagonal of the confusion matrix correspond to correct 
predictions whereas the others imply false positives and false negatives."""

confusion_matrix(y_test, y_pred)
#Accuracy = 1581/3060=51%


#Reintroduce outliers
# Creates pandas DataFrame without the null values. 
to_class = outliers[['Salary','CMV','Count','Claims Rate','Children','Work','Car','Home','Area', 'Life', 'Health','YearsofPol','TotalPremium']].dropna()


# Classify these new elements
knn.predict(to_class)

######################################################################################## Decision tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=0,
                             max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X,     #independents
              y)        #dependents

clf.feature_importances_

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 


dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=list(X.columns),
                                class_names = ['class_' + str(x) for x in np.unique(y)],
                                filled=True,
                                rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)

"""Check veracity of decision tree"""
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)

#Accuracy = 2154/3060=70%

#Introduce new customers
# Creates pandas DataFrame without the null values. 
to_class = outliers[['Salary','CMV','Count','Claims Rate','Children','Work','Car','Home','Area', 'Life', 'Health','YearsofPol','TotalPremium']].dropna()

# Classify these new elements
pd.DataFrame(clf.predict(to_class))

#After we decide to use decision tree we proceed to add the outliers back in our dataset final
outliers = outliers[['ID','Salary','CMV','Count','Claims Rate','Children','Work','Car','Home','Area','Edu','Life', 'Health','YearsofPol','TotalPremium']].dropna()
outliers['FinalCluster'] = pd.DataFrame(clf.predict(to_class))

cleanfinal=clust_merged_full.append(outliers, ignore_index=True)
cleanfinal.to_excel("FinalLabelledCustomerlist.xlsx")

###########################################
'''INTERPRETING CLUSTERS IN VISUALS'''
##########################################

#Dataframes with avg values of population versus clusters
x=policies_scaled_clusters.transpose()
population=cleanstats.transpose()
x=x.merge(population[['mean']],right_index=True,left_index=True)


y=custnumclusters.transpose()
population=cleanstats.transpose()
y=y.merge(population[['mean']],right_index=True,left_index=True)

z=cat_centroids.transpose()
population=cleanstats.transpose()
z=z.merge(population[['50%']],right_index=True,left_index=True)

#Clustered barchart to show standardized averages for each cluster
#show custnum clusters
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # # Setting all variables between 0 and 1 in order to better visualize the results
q=clust_merged_full.drop(columns=['Children','Area','Edu','ID','PolLabels','CustNumLabels','FinalCluster'])
barchart_val = pd.DataFrame(scaler.fit_transform(q))
barchart_val.columns=q.columns


barchart_val['CustNumLabels'] = clust_merged_full['CustNumLabels']
tidy1 = barchart_val.melt(id_vars='CustNumLabels')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='CustNumLabels', y='value', hue='variable',data=tidy1, palette='Set3')
plt.legend(loc='upper right')
# plt.savefig("mess.jpg", dpi=300)
plt.savefig("barchart custnum.jpg", dpi=300)

#show policies clusters
scaler = MinMaxScaler() # # Setting all variables between 0 and 1 in order to better visualize the results
q=clust_merged_full.drop(columns=['Children','Area','Edu','ID','PolLabels','CustNumLabels','FinalCluster'])
barchart_val = pd.DataFrame(scaler.fit_transform(q))
barchart_val.columns=q.columns


barchart_val['PolLabels'] = clust_merged_full['PolLabels']
tidy1 = barchart_val.melt(id_vars='PolLabels')
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='PolLabels', y='value', hue='variable',data=tidy1, palette='Set3')
plt.legend(loc='upper right')
# plt.savefig("mess.jpg", dpi=300)
plt.savefig("barchart policies.jpg", dpi=300)

#show all 6 micro-segments
scaler = MinMaxScaler() # Setting all variables between 0 and 1 in order to better visualize the results
q=clust_merged_full.drop(columns=['Children','Area','Edu','ID','FinalCluster','CustNumLabels','PolLabels'])
barchart_val = pd.DataFrame(scaler.fit_transform(q))
barchart_val.columns=q.columns


barchart_val['FinalCluster'] = clust_merged_full['FinalCluster']
tidy1 = barchart_val.melt(id_vars='FinalCluster') # formatting for barchart
fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(x='FinalCluster', y='value', hue='variable',data=tidy1, palette='Set3')
plt.legend(loc='upper right')
# plt.savefig("mess.jpg", dpi=300)
plt.savefig("barchart finalcluster.jpg", dpi=300)
