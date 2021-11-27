import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import rankdata
import seaborn as sns
# read all files from csv
education = pd.read_csv("/Users/shauryagaur/Desktop/revelio/education.csv") # contains user_id, jobtitle, start_date, end_date
position = pd.read_csv("/Users/shauryagaur/Desktop/revelio/positions.csv") # contains user_id, jobtitle, start_date, end_date, major
df1 = pd.to_datetime(position['startdate'])
df2 = pd.to_datetime(position['enddate'])
diff = df2 - df1
diff = diff.astype('timedelta64[D]')
diff.dropna(how = 'any')
position['days'] = diff
df3 = pd.to_datetime(education['startdate'])
df4 = pd.to_datetime(education['enddate'])
diff1 = df4 - df3
diff1 = diff1.astype('timedelta64[D]')
diff1.dropna(how = 'any')
education['days'] = diff1
job_title = pd.read_csv("/Users/shauryagaur/Desktop/revelio/jobtitle_seniority.csv") # seniority, jobtitle, user_id, jobtitle, start_date, end_date
# Creating a data set with all the common tables
df = pd.merge(education, position, on = ['user_id', 'days'], how = 'inner')
df_final = pd.merge(df, job_title, on = ['user_id'], how = 'inner')
# Plotting seniority versus days for 100 points for groundtruth
hi2 = round(df_final['seniority'][:100],2) # Sorry for the weird variable names!
hi1=df_final['days'][:100]
sns.regplot(x=rankdata(hi1), y =rankdata(hi2), fit_reg = True)
plt.xlabel("Days")
plt.ylabel("Seniority")
plt.title("Days vs. seniority")
plt.show()
# using linear regression to find age as ground truth
import statsmodels.formula.api as smf
age = df_final['seniority']*4 + 10 # ground truth
df_final['age'] = age
model = smf.ols('age ~ seniority', data=df_final)
model = model.fit()
age_pred = model.predict()
# Plot regression against ground truth
plt.figure(figsize=(12, 6))
plt.plot(df_final['age'], df_final['seniority'], 'o')  # scatter plot showing actual data
plt.plot(df_final['age'].dropna(), age_pred, 'r')   # regression line
plt.xlabel('age')
plt.ylabel('seniority')
plt.title('Age vs Seniority')
plt.show() # Regression plot
# ======= Prediction using training and testing =============
from sklearn.model_selection import train_test_split
# Cleaning up the dataset
df_final.drop(columns=['major'])
X = df_final['seniority']
y = df_final['age']
X = X.values.reshape(-1,1)
# Splitting the dateset intro training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Using random forest classifier to predict the user's age
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
# Fitting the regressor on the training set
regressor.fit(X_train, y_train)
# Predicting the user's age
Y_pred = regressor.predict(X_test)
# Appending the relevant columns to our final table
age = df_final['age']
data = age.isin(y_test)
df_final['age predicted'] = data
print(df_final) # Our final dataset :)

