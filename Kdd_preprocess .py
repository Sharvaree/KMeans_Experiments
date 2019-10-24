#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[79]:


Col_names=["duration","sym_protocol_type", "sym_service", "sym_flag", "src_bytes", 
           "dst_bytes","sym_land", "wrong_fragment", "urgent", "hot", 
           "num_failed_logins", "sym_logged_in", "num_compromised", "rootshell" ,"su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
           "sym_host_login" ,"sym_guest_login","count", "srv_count", "serror_rate", 
           "srv_serror_rate", "rerror_rate", "srv_rerror_rate","same_srv_rate", "dif_srv_rate", 
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count","dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
data = pd.read_csv('kddcup.data_10_percent_corrected', sep=",", names= Col_names)
len(Col_names)


# In[81]:


Num_data= data.drop(["sym_protocol_type", "sym_service", "sym_flag", "sym_land", "sym_logged_in", 
           "sym_host_login", "sym_guest_login", "dst_host_count", "dst_host_srv_count",
                     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "rootshell" ,"su_attempted",
                     "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"], axis=1)


# In[82]:


Column_names= list(Num_data.columns.values)
Header= Column_names[0:22]
df_label=Num_data[["label"]]


# In[86]:


Data_processed=np.zeros((494021, 22))
min_max_scaler = preprocessing.MinMaxScaler()
for i in range(len(Column_names)-1):
    Col= data[[Column_names[i]]].values.astype(float)
    #print(Col.shape)
    normalized_array = min_max_scaler.fit_transform(Col)
    #print(normalized_array.shape)
    Data_processed[:,i]= np.squeeze(normalized_array)


# In[87]:


np.savetxt("kdd_processed.csv", Data_processed, delimiter=" ")


# In[88]:


data_processed = pd.read_csv('kdd_processed.csv', sep=" ", names= Header)

#add labels
df_data = pd.concat([df_label,data_processed], axis=1)


# In[89]:


Label_names= df_data.label.unique()
print(Label_names)


# In[90]:


#create unique list of names
unique_labels = df_data.label.unique()

#create a data frame dictionary to store your data frames
data_processed_dict = {elem : pd.DataFrame for elem in unique_labels}

for key in data_processed_dict.keys():
    data_processed_dict[key] = df_data[:][df_data.label == key]


# In[123]:


num_points=[]
for i in range(len(Label_names)-1):
    num_points.append(len(data_processed_dict[Label_names[i]]))
    
num_points


# In[98]:


Cluster_one= data_processed_dict["normal."]
C1_mean=Cluster_one.mean().values.reshape(1,-1)
print(C1_mean.shape)
C1_points= Cluster_one.values


# In[101]:


C1_points[:,1:23].shape


# In[102]:


C1_dif=np.subtract(C1_points[:,1:23],C1_mean)
C1_distance= np.sum(C1_dif**2, axis=1)
print(np.mean(C1_distance))


# In[103]:


plt.plot(C1_distance)
np.histogram(C1_distance, bins=[0,2,4,6, 8, 10])


# In[43]:


#plt.hist(C1_distance, bins=[0,2,4,6])


# In[104]:


Cluster_two= data_processed_dict["neptune."]
C2_mean=Cluster_two.mean().values.reshape(1,-1)
C2_points= Cluster_two.values
C2_dif=np.subtract(C2_points[:,1:23],C2_mean)
C2_distance= np.sum(C2_dif**2, axis=1)
print(np.mean(C2_distance))
plt.plot(C2_distance)
np.histogram(C2_distance, bins=[0,2,4,6,8,10])


# In[118]:


C2_points.shape


# In[105]:


np.histogram(C2_distance, bins=[0,1,2,3, 4,5, 6,7, 8,9, 10])


# In[108]:


Cluster_three= data_processed_dict["smurf."]


# In[112]:


C3_points= Cluster_three.values
C3_points[:,1:23]


# In[113]:


C3_mean=Cluster_three.mean().values.reshape(1,-1)


# In[119]:


C3_points.shape


# In[114]:


C3_points= Cluster_three.values
C3_dif=np.subtract(C3_points[:,1:23],C3_mean)
C3_distance= np.sum(C3_dif**2, axis=1)

plt.plot(C3_distance)
np.histogram(C3_distance, bins=[0,2,5,6,8, 10])


# In[115]:


#average distance of all points from cluster mean
print('C1',np.mean(C1_distance))
print('C2',np.mean(C2_distance))
print('C3',np.mean(C3_distance))


# In[116]:


#Distance between cluster means
C1_C2=np.sum((C1_mean-C2_mean)**2)
print('C1_C2',C1_C2)
C1_C3=np.sum((C1_mean-C3_mean)**2)
print('C1_C3',C1_C3)
C3_C2=np.sum((C3_mean-C2_mean)**2)
print('C3_C2', C3_C2)


# In[124]:


Cluster_four= data_processed_dict["back."]
C4_mean=Cluster_four.mean().values.reshape(1,-1)
C4_points= Cluster_four.values
C4_dif=np.subtract(C4_points[:,1:23],C4_mean)
C4_distance= np.sum(C4_dif**2, axis=1)
print(np.mean(C4_distance))
plt.plot(C4_distance)
np.histogram(C4_distance, bins=[0,2,4,6,8,10])


# In[129]:


C1_C4=np.sum((C1_mean-C4_mean)**2)
print(C1_C4)
C2_C4=np.sum((C2_mean-C4_mean)**2)
print(C2_C4)


# In[127]:


Cluster_five= data_processed_dict["rootkit."]
C5_mean=Cluster_five.mean().values.reshape(1,-1)
C5_points= Cluster_five.values
C5_dif=np.subtract(C5_points[:,1:23],C4_mean)
C5_distance= np.sum(C5_dif**2, axis=1)
print(np.mean(C5_distance))
plt.plot(C5_distance)
np.histogram(C5_distance, bins=[0,2,4,6,8,10])


# In[130]:


C1_C5=np.sum((C1_mean-C5_mean)**2)
print(C1_C5)
C2_C5=np.sum((C2_mean-C5_mean)**2)
print(C2_C5)


# In[ ]:




