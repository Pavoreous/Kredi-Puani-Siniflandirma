#!/usr/bin/env python
# coding: utf-8

# ## About Dataset 
# ### [Problem Statement](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
# > **You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.**
# 
# ### Task
# **Given a person’s credit-related information, build a machine learning model that can classify the credit score.**

# * **ID**         : Represents a unique identification of an entry
# * **Customer_ID**           : Represents a unique identification of a person
# * **Month**             : Represents the month of the year
# * **Name**               : Represents the name of a person
# * **Age**                 : Represents the age of the person
# * **SSN**                   : Represents the social security number of a person
# * **Occupation**            : Represents the occupation of the person
# * **Annual_Income**         : Represents the annual income of the person
# * **Monthly_Inhand_Salary** : Represents the monthly base salary of a person
# * **Num_Bank_Accounts**     : Represents the number of bank accounts a person holds

# ### Credit score Data Modeling
# 
# * [importing the libraries](#importing-the-libraries)
# * [Reading the data](#Reading-the-data)
# * [Exploring the data](#Exploring-the-data)
# * [Edit columns and Data Type](#Edit-columns-and-Data-Type)
# * [Missing data](#Missing-data)
# * [Detect Outliers and Fill NaN Values for Every columns](#Detect-Outliers-and-Fill-NaN-Values-for-Every-columns)
# * [SSN](#SSN)
# * [Monthly_Inhand_Salary](#Monthly_Inhand_Salary)
# * [Num_of_Delayed_Payment](#Num_of_Delayed_Payment)
# * [Changed_Credit_Limit](#Changed_Credit_Limit)
# * [Num_Credit_Inquiries](#Num_Credit_Inquiries)
# * [Credit_History_Age](#Credit_History_Age)
# * [Amount_invested_monthly](#Amount_invested_monthly)
# * [Monthly_Balance](#Monthly_Balance)
# * [Occupation](#Occupation)
# * [Type_of_Loan](#Type_of_Loan)
# * [Credit_Mix](#Credit_Mix)
# * [Payment_Behaviour](#Payment_Behaviour)
# * [Age](#Age)
# * [Annual_Income](#Annual_Income)
# * [Num_Bank_Accounts](#Num_Bank_Accounts)
# * [Num_Credit_Card](#Num_Credit_Card)
# * [Interest_Rate](#Interest_Rate)
# * [Num_of_Loan](#Num_of_Loan)
# * [Delay_from_due_date](#Delay_from_due_date)
# * [Outstanding_Debt](#Outstanding_Debt)
# * [Credit_Utilization_Ratio](#Credit_Utilization_Ratio)
# * [Total_EMI_per_month](#Total_EMI_per_month)
# * [Save process DATA to CSV](#Save-process-DATA-to-CSV)
# * [Drop unimportant columns](#Drop-unimportant-columns)
# * [Encoding categorical features](#Encoding-categorical-features)
# * [Scaling and Split the data](#Scaling-and-Split-the-data)
# * [Model]((#Model)
# * [Logistic Regression](#Logistic-Regression)
# * [KNN](#KNN)
# * [Decision Tree](#Decision-Tree)
# * [Random forest](#Random-forest)
# * [XGBOOST](#XGBOOST)
# * [adaboost](#adaboost)
# * [Voting](#Voting)
# * [compersion between models](#compersion-between-models)

# ## importing the libraries

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn .metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn .metrics import accuracy_score
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


# ## Reading the data

# In[2]:


pd.set_option("display.max.columns", None)

df = pd.read_csv("D:\\Samsung Innovation Campus\\Data\\bank\\train.csv",sep = "," , encoding = "utf-8")
df_test = pd.read_csv("D:\\Samsung Innovation Campus\\Data\\bank\\test.csv",sep = "," , encoding = "utf-8")


# ## Exploring the data

# In[3]:


df.shape , df_test.shape


# In[4]:


df.sample(3)


# In[5]:


df.info()


# In[6]:


df.describe().T.style.background_gradient(cmap='Blues').set_properties(**{'font-family':'Segoe UI'})


# In[7]:


df.describe(exclude=np.number).T.style.background_gradient(cmap='Blues').set_properties(**{'font-family':'Segoe UI'})


# In[8]:


missing_values_df=df.isna().sum()
missing_values_df


# In[9]:


df_na = (missing_values_df / len(df)) * 100

# drop columns without missing values 
df_na = df_na.drop(df_na[df_na == 0].index)

#sort
df_na=df_na.sort_values(ascending=False)

# create plot
f, ax = plt.subplots(figsize=(9, 6))
plt.xticks(rotation='45')
sns.barplot(x=df_na.index, y=df_na)
ax.set(title='Percentage of missing data by feature', ylabel='Percentage missing')
plt.show()


# In[10]:


df.columns


# In[11]:


object_columns=list(df.select_dtypes(include='object').columns)


# In[12]:


df[object_columns].head()


# In[13]:


df.shape


# ## Edit columns and Data Type

# In[14]:


df = df.applymap(lambda x: x if x is np.NaN or not isinstance(x, str) else str(x).strip('_ ,"')).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN)


# In[15]:


df.head()


# In[16]:


df['ID'] = df.ID.apply(lambda x: int(x, 16))


# In[17]:


df['Customer_ID'] = df.Customer_ID.apply(lambda x: int(x[4:], 16))


# In[18]:


df['Age'] = df.Age.astype(int)        


# In[19]:


df['SSN'] = df.SSN.apply(lambda x: x if x is np.NaN else float(str(x).replace('-', ''))).astype(float)


# In[20]:


df['Annual_Income'] = df.Annual_Income.astype(float)


# In[21]:


df['Num_of_Loan'] = df.Num_of_Loan.astype(int) 


# In[22]:


df['Num_of_Delayed_Payment'] = df.Num_of_Delayed_Payment.astype(float)


# In[23]:


df['Changed_Credit_Limit'] = df.Changed_Credit_Limit.astype(float)


# In[24]:


df['Outstanding_Debt'] = df.Outstanding_Debt.astype(float)


# In[25]:


df['Amount_invested_monthly'] = df.Amount_invested_monthly.astype(float)


# In[26]:


df['Monthly_Balance'] = df.Monthly_Balance.astype(float)


# In[27]:


def Month_Converter(x):
    if pd.notnull(x):
        num1 = int(x.split(' ')[0])
        num2 = int(x.split(' ')[3])
      
        return num1*12+num2
    else:
        return x


# In[28]:


df['Credit_History_Age'] = df.Credit_History_Age.apply(lambda x: Month_Converter(x)).astype(float)


# In[29]:


df.shape


# In[30]:


object_columns=list(df.select_dtypes(include='object').columns)
df[object_columns].head()


# In[31]:


num_columns = list(df.select_dtypes(include=["int64","float64"]).columns)
df[num_columns].head()


# ## Missing data

# In[32]:


def columns_with_missing_values(DataFrame):
    missing_columns=(DataFrame.isnull().sum())
    return missing_columns[missing_columns > 0]
columns_with_missing_values(df)


# In[33]:


missing_columns=(df.isnull().sum())
(missing_columns[missing_columns > 0]).index


# In[34]:


miss_num_columns = list(df[(missing_columns[missing_columns > 0]).index].select_dtypes(include=["int64","float64"]).columns)
miss_object_columns=list(df[(missing_columns[missing_columns > 0]).index].select_dtypes(include='object').columns)


# In[35]:


df.shape


# ## Detect Outliers and Fill NaN Values for Every columns

# In[36]:


miss_num_columns = list(df[(missing_columns[missing_columns > 0]).index].select_dtypes(include=["int64","float64"]).columns)
miss_num_columns


# In[1]:


def Distribution2(columne,data,i):
    fig, ax = plt.subplots(1,2, figsize = (15,5))
    font_dict = {'fontsize': 14}
    title=['Before Distribution','After Distribution']
    ax = np.ravel(ax)
    if i==1:
        sns.set(style='whitegrid')
        sns.kdeplot(data=data,x=columne ,ax = ax[0],color='r').set_title(title[i])
        sns.boxplot(data=data,x=columne ,ax = ax[1],palette='magma').set_title(title[i])
    else:
        sns.set(style='whitegrid')
        sns.kdeplot(data=data,x=columne ,ax = ax[0],color='#2171b5').set_title(title[i])
        sns.boxplot(data=data,x=columne ,ax = ax[1],color='#2171b5').set_title(title[i])
        
    ax = np.reshape(ax, (1, 2))
    plt.tight_layout()


# In[38]:


data=df.copy()


# ### SSN

# In[39]:


data.drop('SSN',axis=1,inplace=True)


# In[40]:


data.shape


# ### Monthly_Inhand_Salary

# In[41]:


Distribution2(columne='Monthly_Inhand_Salary',data=data,i=0)


# In[42]:


def get_Monthly_Inhand_Salary(row):
    if pd.isnull(row['Monthly_Inhand_Salary']):
        Monthly_Inhand_Salary=(data[data['Customer_ID']==row['Customer_ID']]['Monthly_Inhand_Salary'].dropna()).mode()
        try:
            return Monthly_Inhand_Salary[0]
        except:
            return np.NaN
    else:
        return row['Monthly_Inhand_Salary']


# In[43]:


data['Monthly_Inhand_Salary']=data.apply(get_Monthly_Inhand_Salary,axis=1)


# In[44]:


#Detect Outliers
print(data[data['Monthly_Inhand_Salary']>= 13500].shape)
data=data[data.Monthly_Inhand_Salary < 13500]


# In[45]:


data.shape


# In[46]:


Distribution2(columne='Monthly_Inhand_Salary',data=data,i=1)


# ### Num_of_Delayed_Payment

# In[47]:


Distribution2(columne='Num_of_Delayed_Payment',data=data,i=0)


# In[48]:


def get_Num_of_Delayed_Payment(row):
    if pd.isnull(row['Num_of_Delayed_Payment']):
        Num_of_Delayed_Payment=(data[data['Customer_ID']==row['Customer_ID']]['Num_of_Delayed_Payment'].dropna()).mode()
        try:
            return Num_of_Delayed_Payment[0]
        except:
            return np.NaN
    else:
        return row['Num_of_Delayed_Payment']


# In[49]:


data['Num_of_Delayed_Payment']=data.apply(get_Num_of_Delayed_Payment,axis=1)


# In[50]:


print(data[data['Num_of_Delayed_Payment']>=150].shape)
print(data[data['Num_of_Delayed_Payment'] < 0].shape)
data=data[data['Num_of_Delayed_Payment']< 150]
data=data[data['Num_of_Delayed_Payment'] >= 0]


# In[51]:


data.shape


# In[52]:


Distribution2(columne='Num_of_Delayed_Payment',data=data,i=1)


# ### Changed_Credit_Limit

# In[53]:


Distribution2(columne='Changed_Credit_Limit',data=data,i=0)


# In[54]:


def get_Changed_Credit_Limit(row):
    if pd.isnull(row['Changed_Credit_Limit']):
        Changed_Credit_Limit=(data[data['Customer_ID']==row['Customer_ID']]['Changed_Credit_Limit'].dropna()).mode()
        try:
            return Changed_Credit_Limit[0]
        except:
            return np.NaN
    else:
        return row['Changed_Credit_Limit']


# In[55]:


data['Changed_Credit_Limit']=data.apply(get_Changed_Credit_Limit,axis=1)


# In[56]:


print(data[data['Changed_Credit_Limit']>=30].shape)
data=data[data['Changed_Credit_Limit'] < 30]


# In[57]:


data.shape


# In[58]:


Distribution2(columne='Changed_Credit_Limit',data=data,i=1)


# ### Num_Credit_Inquiries

# In[59]:


Distribution2(columne='Num_Credit_Inquiries',data=data,i=0)


# In[60]:


def get_Num_Credit_Inquiries(row):
    if pd.isnull(row['Num_Credit_Inquiries']):
        Num_Credit_Inquiries=(data[data['Customer_ID']==row['Customer_ID']]['Num_Credit_Inquiries'].dropna()).mode()
        try:
            return Num_Credit_Inquiries[0]
        except:
            return np.NaN
    else:
        return row['Num_Credit_Inquiries']


# In[61]:


data['Num_Credit_Inquiries']=data.apply(get_Num_Credit_Inquiries,axis=1)


# In[62]:


print(data[data['Num_Credit_Inquiries']>=50].shape)
data=data[data['Num_Credit_Inquiries']<50]


# In[63]:


data.shape


# In[64]:


Distribution2(columne='Num_Credit_Inquiries',data=data,i=1)


# ### Credit_History_Age

# In[65]:


Distribution2(columne='Credit_History_Age',data=data,i=0)


# In[66]:


def get_Credit_History_Age(row):
    if pd.isnull(row['Credit_History_Age']):
        Credit_History_Age=(data[data['Customer_ID']==row['Customer_ID']]['Credit_History_Age'].dropna()).mode()
        try:
            return Credit_History_Age[0]
        except:
            return np.NaN
    else:
        return row['Credit_History_Age']


# In[67]:


data['Credit_History_Age']=data.apply(get_Credit_History_Age,axis=1)


# In[68]:


Distribution2(columne='Credit_History_Age',data=data,i=1)


# ### Amount_invested_monthly

# In[69]:


Distribution2(columne='Amount_invested_monthly',data=data,i=0)


# In[70]:


def get_Amount_invested_monthly(row):
    if pd.isnull(row['Amount_invested_monthly']):
        Amount_invested_monthly=(data[data['Customer_ID']==row['Customer_ID']]['Amount_invested_monthly'].dropna()).mode()
        try:
            return Amount_invested_monthly[0]
        except:
            return np.NaN
    else:
        return row['Amount_invested_monthly']


# In[71]:


data['Amount_invested_monthly']=data.apply(get_Amount_invested_monthly,axis=1)


# In[72]:


print(data[data['Amount_invested_monthly']>=1000].shape)
data=data[data['Amount_invested_monthly']<1000]


# In[73]:


data.shape


# In[74]:


Distribution2(columne='Amount_invested_monthly',data=data,i=1)


# ### Monthly_Balance

# In[75]:


Distribution2(columne='Monthly_Balance',data=data,i=0)


# In[76]:


def get_Monthly_Balance(row):
    if pd.isnull(row['Monthly_Balance']):
        Monthly_Balance=(data[data['Customer_ID']==row['Customer_ID']]['Monthly_Balance'].dropna()).mode()
        try:
            return Monthly_Balance[0]
        except:
            return np.NaN
    else:
        return row['Monthly_Balance']


# In[77]:


data['Monthly_Balance']=data.apply(get_Monthly_Balance,axis=1)


# In[78]:


print(data[data['Monthly_Balance'] <= 0].shape)
data = data[data['Monthly_Balance'] > 0]


# In[79]:


data.shape


# In[80]:


Distribution2(columne='Monthly_Balance',data=data,i=1)


#  ____

# In[81]:


missing_columns=data.isnull().sum()
miss_num_columns = list(data[(missing_columns[missing_columns > 0]).index].select_dtypes(include=["int64","float64"]).columns)
miss_num_columns


# ---

# In[82]:


columns_with_missing_values(data)


# In[83]:


miss_object_columns=list(df[(missing_columns[missing_columns > 0]).index].select_dtypes(include='object').columns)
miss_object_columns


# ### Occupation

# In[84]:


def get_Occupation(row):
    if pd.isnull(row['Occupation']):
        Occupation=(data[data['Customer_ID']==row['Customer_ID']]['Occupation'].dropna()).mode()
        try:
            return Occupation[0]
        except:
            return np.NaN
    else:
        return row['Occupation']


# In[85]:


data['Occupation']=data.apply(get_Occupation,axis=1)


# In[86]:


data[data['Occupation'].isnull()]


# In[87]:


data['Occupation'] = data['Occupation'].fillna(data['Occupation'].mode()[0])


# In[88]:


len(data[data['Occupation'].isnull()])


# ### Type_of_Loan

# In[89]:


data.head(2)


# In[90]:


data['Type_of_Loan'] = data['Type_of_Loan'].fillna('Not Specified')


# In[91]:


def get_Diff_Values_Colum(df_data):
    valu=['Auto Loan','Credit-Builder Loan','Debt Consolidation Loan','Home Equity Loan','Mortgage Loan','Not Specified',
          'Payday Loan','Personal Loan','Student Loan']
    for x in valu:
        df_data[x] = np.NAN
        
    index=0
    for i in df_data['Type_of_Loan']:
        diff_value=[]
        if  ',' not in i:
            diff_value.append(i.strip())
        else:
            for data in map(lambda x:x.strip(), i.replace('and','').split(',')):
                if not data in diff_value:
                    diff_value.append(data)
        
        for x in valu:
            if x in diff_value:
                df_data[x].iloc[index]=1
        index=index+1
        
    for x in valu:
        df_data[x] = df_data[x].fillna(0)
        df_data[x] = df_data[x].astype(int) 
    return df_data

data=get_Diff_Values_Colum(data)


# In[92]:


data.drop('Type_of_Loan',axis=1,inplace=True)


# In[93]:


data.head(2)


# ## Credit_Mix

# In[94]:


def get_Credit_Mix(row):
    if pd.isnull(row['Credit_Mix']):
        Credit_Mix=(data[data['Customer_ID']==row['Customer_ID']]['Credit_Mix'].dropna()).mode()
        try:
            return Credit_Mix[0]
        except:
            return np.NaN
    else:
        return row['Credit_Mix']


# In[95]:


data['Credit_Mix']=data.apply(get_Credit_Mix,axis=1)


# In[96]:


data['Credit_Mix'] = data['Credit_Mix'].fillna(data['Credit_Mix'].mode()[0])


# In[97]:


len(data[data['Credit_Mix'].isnull()])


# ## Payment_Behaviour

# In[98]:


def get_Payment_Behaviour(row):
    if pd.isnull(row['Payment_Behaviour']):
        Payment_Behaviour=(data[data['Customer_ID']==row['Customer_ID']]['Payment_Behaviour'].dropna()).mode()
        try:
            return Payment_Behaviour[0]
        except:
            return np.NaN
    else:
        return row['Payment_Behaviour']


# In[99]:


data['Payment_Behaviour']=data.apply(get_Payment_Behaviour,axis=1)


# In[100]:


data['Payment_Behaviour'] = data['Payment_Behaviour'].fillna(data['Payment_Behaviour'].mode()[0])


# In[101]:


len(data[data['Payment_Behaviour'].isnull()])


#  ----

# In[102]:


columns_with_missing_values(data)


# ----

# In[103]:


num_columns = list(data.select_dtypes(include=["int64","float64",'int32']).columns)
num_columns=num_columns[2:-9]


# In[104]:


process=['Monthly_Inhand_Salary','Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
         'Credit_History_Age','Amount_invested_monthly','Monthly_Balance']


# In[105]:


for i in num_columns:
    if i not in process:
        print(i)


# ## Age

# In[106]:


Distribution2(columne='Age',data=data,i=0)


# In[107]:


print(data[data['Age'] > 60].shape)


# In[108]:


def get_age(row):
    if (60 < row['Age']) or (0 > row['Age']) :
        Age=(data[data['Customer_ID']==row['Customer_ID']]['Age'].dropna()).mode()
        try:
            return Age[0]
        except:
            return np.NaN
    else:
        return row['Age']


# In[109]:


data['Age']=data.apply(get_age,axis=1)


# In[110]:


data[data['Age'] > 60].sort_values('Age')


# In[111]:


data.drop(data[data['Age'] > 60].index,axis=0,inplace=True)


# In[112]:


data[data['Age'] < 0].sort_values('Age')


# In[113]:


data.drop(data[data['Age']  < 0].index,axis=0,inplace=True)


# In[114]:


Distribution2(columne='Age',data=data,i=1)


# ## Annual_Income

# In[115]:


Distribution2(columne='Annual_Income',data=data,i=0)


# In[116]:


def get_Annual_Income(row):
    if 150000 < row['Annual_Income'] :
        Annual_Income=(data[data['Customer_ID']==row['Customer_ID']]['Annual_Income'].dropna()).mode()
        try:
            return Annual_Income[0]
        except:
            return np.NaN
    else:
        return row['Annual_Income']


# In[117]:


data['Annual_Income']=data.apply(get_Annual_Income,axis=1)


# In[118]:


data[data['Annual_Income'] > 165000].sort_values('Annual_Income')


# In[119]:


data.drop(data[data['Annual_Income']  > 165000].index,axis=0,inplace=True)


# In[120]:


Distribution2(columne='Annual_Income',data=data,i=1)


# ## Num_Bank_Accounts

# In[121]:


Distribution2(columne='Num_Bank_Accounts',data=data,i=0)


# In[122]:


def get_Num_Bank_Accounts(row):
    if 12 < row['Num_Bank_Accounts'] :
        Num_Bank_Accounts=(data[data['Customer_ID']==row['Customer_ID']]['Num_Bank_Accounts'].dropna()).mode()
        try:
            return Num_Bank_Accounts[0]
        except:
            return np.NaN
    else:
        return row['Num_Bank_Accounts']


# In[123]:


data['Num_Bank_Accounts']=data.apply(get_Num_Bank_Accounts,axis=1)


# In[124]:


data[data['Num_Bank_Accounts'] > 12]


# In[125]:


data.drop(data[data['Num_Bank_Accounts']  > 12].index,axis=0,inplace=True)
data.drop(data[data['Num_Bank_Accounts']  < 0].index,axis=0,inplace=True)


# In[126]:


Distribution2(columne='Num_Bank_Accounts',data=data,i=1)


# ## Num_Credit_Card

# In[127]:


Distribution2(columne='Num_Credit_Card',data=data,i=0)


# In[128]:


data[data['Num_Credit_Card'] > 14]


# In[129]:


def get_Num_Credit_Card(row):
    if 14 < row['Num_Credit_Card'] :
        Num_Credit_Card=(data[data['Customer_ID']==row['Customer_ID']]['Num_Credit_Card'].dropna()).mode()
        try:
            return Num_Credit_Card[0]
        except:
            return np.NaN
    else:
        return row['Num_Credit_Card']


# In[130]:


data['Num_Credit_Card']=data.apply(get_Num_Credit_Card,axis=1)


# In[131]:


data.drop(data[data['Num_Credit_Card']  > 14].index,axis=0,inplace=True)


# In[132]:


Distribution2(columne='Num_Credit_Card',data=data,i=1)


# ## Interest_Rate

# In[133]:


Distribution2(columne='Interest_Rate',data=data,i=0)


# In[134]:


data[data['Interest_Rate'] > 35].sort_values('Interest_Rate')


# In[135]:


def get_Interest_Rate(row):
    if 35 < row['Interest_Rate'] :
        Interest_Rate=(data[data['Customer_ID']==row['Customer_ID']]['Interest_Rate'].dropna()).mode()
        try:
            return Interest_Rate[0]
        except:
            return np.NaN
    else:
        return row['Interest_Rate']


# In[136]:


data['Interest_Rate']=data.apply(get_Interest_Rate,axis=1)


# In[137]:


Distribution2(columne='Interest_Rate',data=data,i=1)


# ## Num_of_Loan

# In[138]:


Distribution2(columne='Num_of_Loan',data=data,i=0)


# In[139]:


def get_Num_of_Loan(row):
    if (8 < row['Num_of_Loan']) or (0 > row['Num_of_Loan']):
        Num_of_Loan=(data[data['Customer_ID']==row['Customer_ID']]['Num_of_Loan'].dropna()).mode()
        try:
            return Num_of_Loan[0]
        except:
            return np.NaN
    else:
        return row['Num_of_Loan']


# In[140]:


data['Num_of_Loan']=data.apply(get_Num_of_Loan,axis=1)


# In[141]:


data.drop(data[data['Num_of_Loan']  < 0].index,axis=0,inplace=True)


# In[142]:


Distribution2(columne='Num_of_Loan',data=data,i=1)


# ## Delay_from_due_date

# In[143]:


Distribution2(columne='Delay_from_due_date',data=data,i=0)


# ## Outstanding_Debt

# In[144]:


Distribution2(columne='Outstanding_Debt',data=data,i=0)


# ## Credit_Utilization_Ratio

# In[145]:


Distribution2(columne='Credit_Utilization_Ratio',data=data,i=0)


# ## Total_EMI_per_month

# In[146]:


Distribution2(columne='Total_EMI_per_month',data=data,i=0)


# In[147]:


data=data[data['Total_EMI_per_month']<5000]


# In[148]:


Distribution2(columne='Total_EMI_per_month',data=data,i=1)


# In[149]:


data.shape


# ## Save process DATA to CSV

# In[150]:


data.head(3)


# In[151]:


data.to_csv(r"Data\\bank_data.csv", index=False)


# In[152]:


process_df= pd.read_csv("Data\\bank_data.csv",sep = "," , encoding = "utf-8")


# In[153]:


process_df.head(3)


# ## Drop unimportant columns

# In[154]:


def drop_columns(DataFrame):
    lazy_list=['ID','Customer_ID','Name']
    DataFrame.drop(lazy_list, axis=1, inplace=True)
drop_columns(process_df)


# ## Encoding categorical features

# In[155]:


process_df['Month'] = process_df['Month'].map({'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12})


# In[156]:


Occupation_le = LabelEncoder()

process_df['Occupation'] = Occupation_le.fit_transform(process_df['Occupation'])
Occupation_le.classes_


# In[157]:


Credit_Mix_le = LabelEncoder()

process_df['Credit_Mix'] = Credit_Mix_le.fit_transform(process_df['Credit_Mix'])
Credit_Mix_le.classes_


# In[158]:


Payment_Behaviour_le = LabelEncoder()

process_df['Payment_Behaviour'] = Payment_Behaviour_le.fit_transform(process_df['Payment_Behaviour'])
Payment_Behaviour_le.classes_


# In[159]:


Payment_of_Min_Amount_le = LabelEncoder()

process_df['Payment_of_Min_Amount'] = Payment_of_Min_Amount_le.fit_transform(process_df['Payment_of_Min_Amount'])
Payment_of_Min_Amount_le.classes_


# ## Scaling and Split the data

# In[161]:


x = process_df.drop('Credit_Score',axis=1)
y = process_df['Credit_Score']


# In[162]:


y_le = LabelEncoder()

y_Encode = y_le.fit_transform(y)
y_le.classes_


# In[163]:


scaler = MinMaxScaler()
x = scaler.fit_transform(x)


# In[164]:


pca = PCA(n_components=0.98)
x_reduced = pca.fit_transform(x)
print("Number of original features is {} and of reduced features is {}".format(x.shape[1], x_reduced.shape[1]))


# ## Model

# In[165]:


evals = dict()
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    evals[str(name)] = [train_accuracy, test_accuracy]
    print("Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100))
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Poor', 'Standard','Good'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    cm_display.plot(ax=ax)

lr = LogisticRegression().fit(X_train_clean, y_train_clean)
evaluate_classification(lr, "Logistic Regression", X_train_clean, X_test_clean, y_train_clean, y_test_clean)
# In[166]:


def feature_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)
    plt.figure(figsize=(15,8))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.title('feature importances for Decision Tree')
    plt.show()


# In[167]:


features = process_df.drop(['Credit_Score'] , axis = 1)


# ## Logistic Regression

# In[168]:


from sklearn.linear_model import LogisticRegression


# In[169]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[170]:


lr = LogisticRegression(C = 100)


# In[171]:


lr.fit(x_train , y_train)


# In[172]:


lr_score_train=lr.score(x_train , y_train)
lr_score_train


# In[173]:


lr_score_test=lr.score(x_test , y_test)
lr_score_test


# In[174]:


evaluate_classification(lr, "Logistic Regression", x_train, x_test, y_train, y_test)


# In[175]:


Y_pred=lr.predict(x_test)


# In[176]:


pd.DataFrame((lr.coef_).T ,process_df.drop('Credit_Score',axis=1).columns ).T


# In[177]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# ## KNN

# In[178]:


from sklearn.neighbors import KNeighborsClassifier


# In[179]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[180]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[181]:


knn.fit(x_train , y_train)


# In[182]:


knn_score_train=knn.score(x_train , y_train)
knn_score_train


# In[183]:


knn_score_test=knn.score(x_test , y_test)
knn_score_test


# In[184]:


evaluate_classification(knn, "KNeighborsClassifiern", x_train,x_test,y_train,y_test)


# In[185]:


Y_pred=knn.predict(x_test)


# In[186]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# ## Decision Tree

# In[187]:


from sklearn.tree  import DecisionTreeClassifier


# In[188]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[189]:


dt =DecisionTreeClassifier(max_features=14 ,    max_depth=12)


# In[190]:


dt.fit(x_train , y_train)


# In[191]:


dt_score_train=dt.score(x_train , y_train)
dt_score_train


# In[192]:


dt_score_test=dt.score(x_test , y_test)
dt_score_test


# In[193]:


evaluate_classification(dt, "DecisionTreeClassifier", x_train,x_test,y_train,y_test)


# In[194]:


Y_pred=dt.predict(x_test)


# In[195]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# In[196]:


from sklearn import tree

#to a simple plot,We used max depth of 4
dtt = DecisionTreeClassifier(max_depth=4)

dtt.fit(x_train, y_train)
fig = plt.figure(figsize=(15,12))
tree.plot_tree(dtt , filled=True)
plt.show()


# In[197]:


feature_importances(abs(dt.feature_importances_), features, top=18)


# ## Random forest

# In[198]:


from sklearn.ensemble import RandomForestClassifier


# In[199]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[200]:


rf = RandomForestClassifier(max_features=15 , max_depth=12)


# In[201]:


rf.fit(x_train , y_train)


# In[202]:


rf_score_train=rf.score(x_train , y_train)
rf_score_train


# In[203]:


rf_score_test=rf.score(x_test , y_test)
rf_score_test


# In[204]:


evaluate_classification(rf, "RandomForestClassifier", x_train,x_test,y_train,y_test)


# In[205]:


Y_pred=rf.predict(x_test)


# In[206]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# In[207]:


feature_importances(abs(rf.feature_importances_), features, top=18)


# ## XGBOOST

# In[208]:


from xgboost import XGBClassifier


# In[209]:


x_train,x_test,y_train,y_test = train_test_split(x,y_Encode, test_size=0.3,random_state = 1234)


# In[210]:


xgb = XGBClassifier(max_depth = 5 , learning_rate = 0.3 , objective = 'binary:logistic' , n_estimators= 5, random_state=42)


# In[211]:


xgb.fit(x_train , y_train)


# In[212]:


xgb_score_train=xgb.score(x_train , y_train)
xgb_score_train


# In[213]:


xgb_score_test=xgb.score(x_test , y_test)
xgb_score_test


# In[214]:


evaluate_classification(xgb, "XGBOOST", x_train,x_test,y_train,y_test)


# In[215]:


feature_importances(abs(xgb.feature_importances_), features, top=18)


# In[216]:


Y_pred=xgb.predict(x_test)


# In[217]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# ## adaboost

# In[218]:


from sklearn.ensemble import AdaBoostClassifier


# In[219]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[220]:


ada = AdaBoostClassifier(n_estimators=8, learning_rate=0.8)


# In[221]:


ada.fit(x_train , y_train)


# In[222]:


ada_score_train=ada.score(x_train , y_train )
ada_score_train


# In[223]:


ada_score_test=ada.score(x_test , y_test)
ada_score_test


# In[224]:


evaluate_classification(ada, "adaboost", x_train,x_test,y_train,y_test)


# In[225]:


feature_importances(abs(ada.feature_importances_), features, top=18)


# In[226]:


Y_pred=ada.predict(x_test)


# In[227]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# ## Voting

# In[228]:


from sklearn.ensemble import VotingClassifier


# In[229]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1234)


# In[230]:


clf1 = LogisticRegression(C = 100)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = DecisionTreeClassifier(max_features=14 ,    max_depth=12)
clf4 = RandomForestClassifier(max_features=15 , max_depth=12)
clf5 = XGBClassifier(max_depth = 5 , learning_rate = 0.3 , objective = 'binary:logistic' , n_estimators= 5, random_state=42)
clf6 = AdaBoostClassifier(n_estimators=8, learning_rate=0.8)


# In[231]:


v_clf = VotingClassifier(estimators=[("LogisticRegression" , clf1) , ('KNeighborsClassifier' , clf2) ,
                                     ("XGBClassifier" , clf5) ,("RandomForestClassifier" , clf4),
                                     ("DecisionTreeClassifier",clf3),("AdaBoostClassifier",clf6)] , voting = "hard")


# In[232]:


v_clf.fit(x_train , y_train)


# In[233]:


v_clf_score_train=v_clf.score(x_train , y_train)
v_clf_score_train


# In[234]:


v_clf_score_test=v_clf.score(x_test , y_test)
v_clf_score_test


# In[235]:


evaluate_classification(v_clf, "Voting", x_train,x_test,y_train,y_test)


# In[236]:


Y_pred=v_clf.predict(x_test)


# In[237]:


data = pd.DataFrame({"Y_test" : y_test , "Y_pred": Y_pred})
data.head(20).T


# ## compersion between models

# In[238]:


models = ['Logistic Regression' , 'KNN' , 'Decision Tree','Random forest','XGBOOST','adaboost','Voting']
data = [[lr_score_train ,lr_score_test ] , [knn_score_train ,knn_score_test ] , [dt_score_train ,dt_score_test ],
       [rf_score_train,rf_score_test],[xgb_score_train,xgb_score_test],[ada_score_train,ada_score_test],
        [v_clf_score_train,v_clf_score_test]]
cols = ["Train score" , "Test score"]
pd.DataFrame(data=data , index= models , columns= cols).sort_values(ascending= False , by = ["Test score","Train score"])


# In[ ]:





# In[ ]:




