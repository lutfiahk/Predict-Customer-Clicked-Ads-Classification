<h1>Predict Customer Clicked Ads Classification by Using Machine Learning</h1>

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/a6a616fa-b00b-403c-99f4-9eaa9dfb9754"/>
</p>

<h4><b><i>by: Lutfia Husna Khoirunnisa</b></i></h4>
<h4><i>Linkedin(https://www.linkedin.com/in/lutfiahk)</i></h4>

<h2>Project Introduction</h2>

A company in Indonesia wants to evaluate the effectiveness of its aired advertisement. This is important for understanding the advertisement's reach and attracting viewers.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/c4986fba-4e46-403e-ba25-756f9cb07a47" width="400" height="300"/>
</p>

<b>50%</b> of customers did not click on the provided advertisement.

So, the objective is to improve the click-through rate of the advertisement. To achieve this, we will analyze historical advertisement data to gain insights and identify patterns that can help us target the right audience.

And we will also develop a machine learning classification model that can effectively identify and reach the most relevant potential customers.

<i>All data models and visualization in this project are built with Python (Pandas, Matplotlib, Plotly express, SKLearn)</i>

We give solutions that identify customer patterns and develop a machine-learning for a classification model that can effectively identify and reach the most relevant potential customers.

<h2>Customers Overview</h2>

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/f31f40db-ee6f-470b-92f9-da67a5d7ae25" width="400" height="300"/>
</p>

It is known that customers who do not click on advertisements tend to be younger, while customers who click on ads tend to be older.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/f31f40db-ee6f-470b-92f9-da67a5d7ae25" width="400" height="300"/>
</p>

It is known that customers who do not click on advertisements tend to spend a longer time on the site, while customers who click on advertisements tend to spend a shorter time on the site.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/7459d5e3-21f9-4d3a-9cdd-d115b31ee9ba" width="400" height="300"/>
</p>

Same as the case of customers' daily internet usage, it is observed that customers who do not click on advertisements tend to have high daily internet usage, while customers who click on ads tend to have low daily internet usage.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/b1ac2722-f2e0-40f2-9e2b-519945ac1d0e" width="400" height="300"/>
</p>

<b>The majority of customers who click on advertisements are adults or elderly individuals with low daily time spent on site.</b>

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/9cad7714-4a49-465c-b0c2-249be1af92d0" width="400" height="300"/>
</p>

This is likely because older customers are more easily influenced by compelling advertisements.

Customers with these characteristics have limited time for site exploration or internet browsing, so they prefer to gather information about a product directly from the given advertisements.

<b>It is known that customers who do not click on advertisements tend to be younger and have high time spent on site.</b>

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/6c094538-a8eb-4771-b14c-ce7bc2c47605" width="400" height="300"/>
</p>

This may be because younger customers are less interested in viewing products only through the given advertisements. They prefer to explore products on their own through the internet or site.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/9e123f10-7999-4626-a874-d8d9b14c8e6f" width="400" height="300"/>
</p>

There is an interesting observation from the graph of income and daily spent on the site shown above, it turns out that customers who do not click on advertisements tend to have both high income and high daily spent on the site.

This may be because customers with higher incomes have the financial means to spend money wisely. They are more interested in gathering information and conducting thorough research before making purchasing decisions, so they spend more time on the site and have higher daily internet usage.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/fe825aa3-d694-41cf-8f7e-33cb068d1870" width="400" height="300"/>
</p>

From the scatter plot of income and age shown above, customers who do not click on advertisements tend to have a younger age with a tendency towards higher income. However, customers who click on advertisements do not exhibit a specific trend or pattern and tend to be more scattered.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/d189d64b-8640-4c1e-be7b-4298952c2d53" width="400" height="300"/>
</p>

Actually, there are no features that exhibit strong correlations among themselves. Therefore, it is somewhat challenging to observe customers' patterns and trends. That's why the presence of a machine learning model will be very helpful in predicting whether customers are more likely to click on advertisements or not.

![image](https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/4b5059de-3d4f-449a-9d34-c10a5384d05d)


Customers who do not click on ads are most active on Friday and Sunday, particularly between 8 PM and 11 PM.
These customers may tend to browse the site during their free time after work, especially on Fridays and Sundays.

Customers who actively click on ads are most active on Sunday, Wednesday, and Thursday between 7 AM and 9 AM. They prefer to browse the site in the morning before starting their activities, and it seems they enjoy opening advertisements while having breakfast, especially on weekends.

<h2>Data Overview & Preprocessing</h2>

This project will use a dataset that contains customer data on historical clicks on advertisements, including the number of customers, age, income, daily internet usage, and more. The dataset consists of <b>1000 records and 11 features.</b>

The dataset is not clean yet as it contains missing values, incorrect values, and unused records that require handling.

<b>DATA PREPROCESSING</b>

DATA CLEANSING :
 
* Dropping unused columns such as 'Unnamed:
0', 'Timestamp', 'City', and 'province' columns, and changing the name of the 'Male' column with 'gender'
* Handling missing values in the 'Daily Time Spent on Site', 'Income', 'Daily Internet Usage', and 'Gender' column

FEATURE ENGINEERING : 
Transform object value with a numerical value, such as :
* Replace 'Laki-laki' with 0, and 'Perempuan' with 1 in the Gender column
* Replace Yes with 1, and No with 0 in the Clicked on Ad column 
* Transform category column to several columns by performing one hot encoding

Data Splitting, outlier handling, and feature normalization also will be conducted.
* DATA SPLITTING

    Next, the data will be split into training and testing sets, where the ratio used is 70% for the training data and 30% for the testing data.

* OUTLIER HANDLING

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/29f61c80-7c4f-4494-a5f0-528c94e4e56a" width="400" height="300"/>
</p>

Outlier handling is also performed, but this handling is only done on the train dataset. This is done to prevent any information leakage. The purpose of outlier handling is to address extreme or unusual observations that can negatively impact the model's performance and generalizability.

We only drop the outlier in the Area income column, because the other columns have no extreme outlier. We got :

    * Number of rows before filtering outliers: 700
    * Number of rows after filtering outliers: 689

* FEATURE NORMALIZATION

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/3fcf3f57-dd64-4533-8dde-b18fd5af3756" width="400" height="300"/>
</p>

Then feature normalization is performed, even though other features tend to be skewed, but they are not extreme, so MinMaxScaler will be used.

* CLASS IMBALANCE HANDLING

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/3cdf5927-1245-4cdb-8aa5-7208b0e84952" width="300" height="300"/>
</p>

Since the data does not have an imbalanced class issue, handling is not necessary.

<h2>Data Modeling</h2>

We use several classification methods to predict customers who click on ads.

<i>(The accuracy metric is used to measure the accuracy of predictions because both positive and negative classes are equally important in this case.)</i>

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/428507c5-6343-4fa1-9aed-7973e69f5494"/>
</p>


<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/5190d536-4749-40a9-8cc6-9b8aac1a97da"/>
</p>

Based on the feature importance scores, it is known that <b>"Age" has a high positive correlation with the target label</b>, indicating that as the customer's age increases, they are more likely to click ads.

The features <b>"daily spent time on site" and "daily internet usage" have high negative correlation values with the target label</b>, indicating that as the daily spent time on site and internet usage decrease, the clients are more likely to click ads.

Here is the following the prediction result from the model:

![image](https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/0abebbc5-ed2c-4b6a-84c7-06e16b746757)


<h2>From the data analysis and the obtained model, we can conclude that:</h2>

<b>Customers who do not click on ads tend to be younger with a higher income. </b>

Their daily time spent on the site tend to be high, indicating that they are exposed to a lot of ads and may feel overwhelmed with the displayed ads, so maybe they prefer to search for product information on their own rather than looking at
displayed ads.

<b>So this is our main challenge</b>, If we want to focus on increasing the click-through rate, we need to provide special ads that match with their preferences, so they feel interested and motivated to click on the ads.

Since they spend more time and exposed with many advertisements, we need to reduce their ads couse it may make them feel overwhelmed. 

It's better to provide them with advertisements at the right time, such as on weekends after they finish their work when they are relaxed and likely to be more interested and have more time to view advertisements.


<b>Customers who click on ads tend to be older with a lower income.</b>

Their daily time spent on the site tends to be low, indicating that they are not inclined to explore products on their own and are more easily drawn to advertisements. This is make sense as advertisements often provide something that contains clickbait, and easily captures attention. 

Additionally, these customers have a relatively low income, which means they are more sensitive to promotions and prices, making them more susceptible to advertisements containing promotional information.

<b>If we want to increase revenue, we can focus on these customers.</b> We can provide advertisements that are more match to their needs. Further analysis of their transaction history needs to be conducted so that we can provide the types of products or advertisements that they need the most.

<h3>Potential Impact</h3>

To measure the potential impact, we use the customer data that we used for the test set.

![image](https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/6f16442f-a90e-41e6-bf96-7cb9813fc0bc)


Assuming we provide personalized ads and give the ads on the right time to customers who are predicted not to click on ads, and this treatment can convince half of the customers to change their minds and click on the given ads, the click-through ads rate can increase from 45% to 71%.

<b>Certainly, if they continue to make transactions, it can significantly increase the revenue.</b>
