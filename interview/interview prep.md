# Interview prep
Below are some questions that I have faced during interviews.
------

**- Talk about 1-2 data science projects you feel most proud of.**

The **first project** I would like to talk about is the one where I built a model to try to predict the outcome of NBA games. I really liked working on it, and I imagine it could be relevant to this position.
I was involved in all steps of this projects, so from hypotheses generation, thinking why a team is more likely to win a game than another, and I decided to do that by looking at team statistics. I wanted to consider both traditional and advanced team statistics, and compare model performance using both types.

Then, I had to extract data off the internet, so I had to look for different sources and decide where to collect the data from. I ended up using Basketball-Reference because the data can be easily collected there. After I had the data in my hands, I had to parse and transform it, and I did that using python, working with pandas and numpy libraries. This involved cleaning the data, removing missing values, and creating new features. I also created some visualizations to help me better understand the data, and for that I used Matplotlib and Seaborn. For example, I looked at the percentage of home team wins across different seasons, which is interesting because play at home is considered an advantage. 

When I felt like I had a good grasp for the dataset, I started working on modeling. So I approached this as a classification problem, considering whether the home team won or not. For that I used different algorithms from the Scikit-Learn package. I was interested in learning more about the algorithms and also check which would give me the best accuracy. I worked with logistic regression, random forest, svm, naive bayes, and XGBoost. To optimize the models, I used GridSearchCV. 

For evaluation, I looked at cross validation and I compared the mean accuracy and standard deviation. Finally, I wanted to try deploying the model on the web, so I built a simple application using Streamlit and deployed it in the Streamlit cloud.

The second project is not related to sports, but I think it is relevant because I worked on it as part of a team. We were tasked with clustering neighborhoods in NYC, and we had complete freedom to decide how we would do that. We decided to do so by looking at healthcare facilities in each area because we thought this would be meaningful and fun topic to look at. 

To start this project, we had to work with different file formats. We had a JSON file with geographical information, so we used pandas and json libraries in python to extract that information. We also worked with Excel files and other csv files, for example containing rent prices and populational information. Numpy, matplotlib and again pandas were useful when doing this. 

We used the requests library to obtain healthcare venue information using the foursquare API. So basically we used the geographical coordinates that we extracted from the JSON file, and then used those as the location where we wanted to get venues from. 

We decided that the most interesting way to create clusters was by looking at the frequency of facility types in each neighborhood. We looked at the top 5 facility types in each neighborhood and used that for clustering. For example, neighborhoods with high numbers of dentists would be clustered together. To do that, we used KMeans clustering. With that data, we were then able to create an actual map of NYC using the folium package. We added markers to the map and color-coded them based on the assigned cluster. 

In this project, it was crucial to have effective communication, so my partner and I met frequently to share our progress, our findings and very importantly our struggles. We use GitHub to share our work, 

**- Why would you pick model A vs model B to solve a particular problem?**
The decision of which model to use depends on the problem at hand. Points we need to consider are the size of the dataset. On the one hand, if the dataset is not very large, we need to consider using models that are more suitable for working with smaller datasets. This could be important in cases where data extraction is not easy or cheap. On the other hand, if the dataset is too large, running a computationally-expensive model may be out of question unless we can run this on the cloud with more powerful machines. It also depends on the complexity of the data, if a simple linear model can already capture the nuances or not. 

**- What is the difference between the different ML evaluation metrics and when would you pick one over the other?**
Metrics are different for regression and classification problems. For **regression:**
- MAE: mean absolute error, calculate the difference between real value and prediction, then calculate the mean. 
- MSE: mean squared error, takes the square of the error instead of the absolute value. This is more prone to inflating the error because of the squaring.
- RMSE: root mean squared error, similar to MSE, but we take the root. It's advantageous because the error is in the same units of the value we try to predict. 

For **classification**:
- Accuracy: correct predictions over total predictions
- Confusion matrix: true positives, false positives, true negatives and false negatives.
- Precision: out of all positives (true and false), how many were actually true positives? looking at correctly identifying the positives.
- Recall: tell us how good the model is at correctly predicting all the positive observations in the dataset.
- F1 score: combination of precision and recall (harmonic mean).


**- What challenges did you face while working on your projects? How did/would you overcome them?**
Data cleaning can be a challenge, because often real world data will have inconsistencies such as missing values, entries different spelling or formatting. With time, I have gotten better at spotting these inconsitencies and addressing them as well. 

Some challenges I faced included being able to communicate my findings to someone who was not working on the project, or has no background in data science. It was the same when I was working in science or in sales - we need to be able to adapt to our audience. Thankfully, during the bootcamp we were able to practice that quite a lot, as we always had to do 5 min presentations talking about our work, our findings, and the challenges we faced.

An obvious challenge is that sometimes I found myself stuck not knowing how to do something with the dataset. This is partly due to lack of experience, so the way I try to address it is by being resourceful, searching on forums like stackoverflow, reading the documentation for the package that I am using, and talking to my peers. 

**- Differences between feature selection and feature engineering.**
In feature selection, we are looking at selecting only the features that we believe are useful in building a model. We can use pairplot from Seaborn, which compares each feature in pairs by doing scatter plots. We are creating thus a subset of the original features.
In feature extraction, we are trying to reduce the number of features by combining different features and remove redundancies in the dataset. So here, we are not creating a subset of original features, but rather creating new ones.

**- How to define the right sample size for your statistical experiments and to spot/mitigate potential biases in your samples?**
The right sample size depends on the dataset you are using. We can try to use our expertise to know how large a dataset should be, for example in sports we don't want often don't want to use data that is from too long ago because a player can change throughout their career, and play styles change across teams over different seasons, so these might not be very large datasets.
The algorithm being used will also influence this choice. For example, nonlinear or non-parametric models usually require larger datasets. 
We can use visualization tools to evaluate if the dataset is biased, for example in a classification problem we can try to address an imbalanced dataset by sampling more from the minority class, or oversampling the majority class. We can also plot the number of samples versus the model performance and see what dataset size is enough to achieve what we need.

**- Explain how algorithm X works under the hood.**
**Logistic regression:** model for binary classification, we use a sigmoid function to fit the data within 0 and 1

**RandomForest:** similar to decision trees, but we combine multiple random decision trees with random features and use the predicted majority class.
	Reduced chance of overfitting because we don't use the same samples every time to build the trees, and not all the same features are trained the same time
	Independet, built in parallel
	
**SVM:** we aim to create a hyperplane to separate the points that belong to each class, for example a surface separating points in a 3D space
	This hyperplane is created taking into account the features, and we aim for a hyperplane that can efficiently separate the points from each class

**Naive Bayes:** assumes that the features are independent, there are different versions, I used Gaussian Naive Bayes 

**XGBoost:** Boosting algorithm, where we train different models sequentially compensating for the errors of the prior model
	Sequential model, so it can take longer

**KMeans**: used for clustering a large set of data points into smaller groups (or clusters). We choose the K number of groups, take the mean of each of these groups, and try to group entries that are not too distant from this mean to that group. 

**DBScan:** can handle nested clusters by looking at density of points. Clusters are in areas with high density, and outliers are in areas with low density.

**Hierarchichal**: build a hierarchy of clusters by looking at similarities at different samples, and then grouping them together.

**- How would you deploy model X into the cloud?**
When I had to deploy a model in the cloud, I exported the selected model as bytes using pickle. Then, we need to write a python script with the application and our model. For example, we could load the pickled model, and then use Flask to write an application that allows a user to communicate with our model. Finally, we run this python script in the cloud, such as in an AWS EC2 server.

**- What makes a good versus a great data scientist?**
A good data scientist has good programming skills, knowledge of the different tools that are available out there. A great data scientist has the curiosity to go beyond just building a model and deploying it, and also asks questions about the meaning of the results that are models provide. This goes beyond just being good at coding, but also involves having a passion for the topic at hand, so in this case knowing and being passionate about sports and sports statistics. Being proactive, trying to learn from different people about the business, and not just the data scientists, conducting investigations without waiting for someone to give directions. 

**- How would you handle conflict and different personalities?**
I think that dialogue is really important when resolving conflict. It's also important to calm down, not take it personally since we are all working on the same team, so keep that in mind when deciding on something that is the best for the business. 

It's important to try to understand the other person's perspective, which can be hard in data science because often you are so deep into the problem that you are trying to solve that it takes a while to pick up what the other side is saying. We need to also give others space so they can express themselves in their own way, and not force our will on other members of the team. 

**- Why do you want to work for Sports IQ specifically?**
Well, I'm very excited about working with sports analytics, and at Sports IQ your team is using machine learning to track games and player performance, which sounds very exciting. From my research, it looks like the company is growing, which is a good sign, and I also noticed you have several data scientists, so it would be a great experience for me to join a data science team and learn from people with more experience. I also like that this looks like a challenging environment, because sports change constantly, and as such the models need to evolve and constantly be reevaluated, and that the team seems to be quite diverse 

**- Why the sports-betting field in general?**
Now that sports betting is being regulated in legal in many countries, I see it as a big market with great potential for growth not just in north america, but in other regions as well. So this is exciting because it is a hot and expanding field combined with something that I'm passionate about, which is sports


====================================**QUESTIONS TO ASK THEM**========================================


My questions:
What projects would be under the scope of this position?

Are you covering only North American leagues? I noticed there is some modeling for the MLS, were you planning to work with European soccer as well? 

What is the routine like for a data scientist/ in the business intelligence team at Sports IQ? 

What is the tech stack at this company? Specifically for data scientists

What are the first months like for a new employee? Is there a training period? 

postgresql
apache airflow
google cloud 
big query
