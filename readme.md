<h1>Predict Customer Clicked Ads Classification by Using Machine Learning</h1>

<h4><b><i>by: Lutfia Husna Khoirunnisa</b></i></h4>
<h4><i>Linkedin(https://www.linkedin.com/in/lutfiahk)</i></h4>

<h2>Project Introduction</h2>

A company in Indonesia wants to evaluate the effectiveness of their aired advertisement. This is important for understanding the advertisement's reach and attracting viewers.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/c4986fba-4e46-403e-ba25-756f9cb07a47" width="400" height="300"/>
</p>

<b>50%</b> of customers did not click on the provided advertisement.

So, the objective is to improve the click-through rate of the advertisement. To achieve this, we will analyze historical advertisement data to gain insights and identify patterns that can help us target the right audience.

And we will also develop a machine learning classification model that can effectively identify and reach the most relevant potential customers.

<i>All data model and visualization in this project are built with Python (Pandas, Matplotlib, Plotly express, SKLearn)</i>

We give solutions which identify customer's patterns and develop a machine learning for classification model that can effectively identify and reach the most relevant potential customers.

<h2>Customers Overview</h2>

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/f31f40db-ee6f-470b-92f9-da67a5d7ae25" width="400" height="300"/>
</p>

It is known that customers who do not click on advertisements tend to be younger, while customers who click on advertisements tend to be older.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/f31f40db-ee6f-470b-92f9-da67a5d7ae25" width="400" height="300"/>
</p>

It is known that customers who do not click on advertisements tend to spend a longer time on the site, while customers who click on advertisements tend to spend a shorter time on the site.

<p align = "center">
   <img src="https://github.com/lutfiahk/Predict-Customer-Clicked-Ads-Classification/assets/99700225/7459d5e3-21f9-4d3a-9cdd-d115b31ee9ba" width="400" height="300"/>
</p>

Same as the case of customer's daily internet usage, it is observed that customers who do not click on advertisements tend to have high daily internet usage, while customers who click on advertisements tend to have low daily internet usage.


<b>The majority of customers who click on advertisements are adults or elderly individuals with low daily time spent on site.</b>


This is likely because older customers are more easily influenced by compelling advertisements.

Customers with these characteristics have limited time for site exploration or internet browsing, so they prefer to gather information about a product directly from the given advertisements.

<b>It is known that customers who do not click on advertisements tend to be younger and have
high time spent on site.</b>

This may be because younger customers are less interested in viewing products only through the given advertisements. They prefer to explore products on their own through the internet or site.

There is an interesting observation from the graph of income and daily spent on the site shown above, it turns out that customers who do not click on advertisements tend to have both high income and high daily spent on the site.

This may be because customers with higher income have the financial means to spend money wisely. They are more interested in gathering information and conducting thorough research before making purchasing decisions, so they spend more time on the site, and have higher daily internet usage.

From the scatter plot of income and age shown above, customers who do not click on advertisements tend to have a younger age with a tendency towards higher income. However, customers who click on advertisements do not exhibit a specific trend or pattern and tend to be more scattered.

Actually, there are no features that exhibit strong correlations among themselves. Therefore, it is somewhat challenging to observe customers patterns and trends. That's why the presence of a machine learning model will be very helpful in predict whether customers are more likely to click on advertisements or not.

Customers who do not click on ads are most active on Friday and Sunday, particularly between 8 PM and 11 PM.
These customers may tend to browse the site during their free time after work, especially on Fridays and Sundays.

Customers who actively click on ads are most active on Sunday, or Wednesday and Thursday between 7 AM and 9 AM. They prefer to browse the site in the morning before starting their activities, and it seems they enjoy opening advertisements while having breakfast, especially on weekends.

<h2>Data Overview & Preprocessing</h2>

This project will use a dataset that contains customers data historical click on advertisement, including the number of customers, age, income, daily internet usage, and more. The dataset consists of <b>1000 records and 11 features.</b>

The dataset is not yet clean as it contains missing values, incorrect values, and unused records that require handling.

<b>DATA PREPROCESSING</b>

DATA CLEANSING :
 
* Droping unused column such as 'Unnamed:
0', 'Timestamp','City', and 'province' columns, and changing the name of 'Male' column with 'gender'
* Handling missing value in the 'Daily Time Spent on Site', 'Income', 'Daily Internet Usage', and 'Gender' column

FEATURE ENGINEERING : 
Transform object value with numerical value, such as :
* Replace 'Laki-laki' with 0, and 'Perempuan' with 1 in the Gender column
* Replace Yes with 1, and No with 0 in the Clicked on Ad column 
* Transfrom category column to several column by performing one hot encoding

Data Splitting, outlier handling, and feature normalization also will be conducted.
* DATA SPLITTING

    Next, the data will be split into training and testing sets, where the ratio used is 70% for the training data and 30% for the testing data.

* OUTLIER HANDLING

    Outlier handling is also performed, but this handling is only done on the train dataset. This is done to prevent any information leakage. The purpose of outlier handling is to address extreme or unusual observations that can negatively impact the model's performance and generalizability.

    We only drop outlier in Area income column, because the other columns have no extreme outlier. We got :

    * Number of rows before filtering outliers: 700
    * Number of rows after filtering outliers: 689

* FEATURE NORMALIZATION

    Then feature normalization is performed, even though other features tend to be skewed, but they are not extreme, so MinMaxScaler will be used.

* CLASS IMBALANCE HANDLING

    Since the data does not have an imbalanced class issue, handling is not necessary.

<h2>Data Modeling</h2>

We use several classification methods to predict customers who click on ads.

<i>(The accuracy metric is used to measure the accuracy of predictions because both positive and negative classes are equally important in this case.)</i>

image.png



Based on the feature importance scores, it is known that <b>"Age" has a high positive correlation with the target label</b>, indicating that as the customer's age increases, they are more likely to click ads.

The features <b>"daily spent time on site" and "daily internet usage" have high negative correlation values with the target label</b>, indicating that as the daily spent time on site and internet usage decrease, the clients are more likely to click ads.

Here is the following the prediction result from the model:


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

Assuming we provide personalized ads and give the ads on the right time to customers who are predicted not to click on ads, and this treatment can convince half of the customers to change their minds and click on the given ads, the click-through ads rate can increase from 45% to 71%.

<b>Certainly, if they continue to make transactions, it can significantly increase the revenue.</b>
