# Project 2: Web APIs & NLP

## Project Statement

This project aims to assist 'redditors' (reddit users) to identify which subreddit (either `r/Science` or `r/Philosophy`) to post their submissions by making a classification model using NLP to classify posts belonging to two different subreddits.

## Summary of Project
According to the [NYTimes](https://archive.nytimes.com/opinionator.blogs.nytimes.com/2012/04/05/philosophy-is-not-a-science/), for roughly 98 percent of the last 2,500 years of Western intellectual history, philosophy was considered the mother of all knowledge. Today, science, not philosophy, has taken up the mantle as the world’s de-facto source of truth, with some no longer sure what philosophy is or is good for anymore. This leads to much confusion, especially in the online space where like-minded individuals gather to discuss topics, and incorrect submissions of topics are found which reduces the effectiveness of these discussion.

Especially in forums like [Reddit](https://www.reddit.com/), where "network of communities can dive into their interests, hobbies and passions", are popular go-to websites for people from different parts of the world to interact in which normally would not happen if not for the streamlined connectivity of the Internet. 

So how can we better maxmise the productivity of these discussions? The best solution is to ensure the correct topics are posted in the correct environment. But due to the confusion, there are bound to be some that are unable to differentiate. This project aims to reduce this problem by attempting to train a model that will be able to identify the differences between posts from each subreddit and classify new posts accordingly through various NLP techniques. The insights gained from our analysis and processing may be useful to the moderators of the respective subreddits and  also normal redditors to know which subreddit their post belongs to.

------------------------------------------------
## Data

### CSV Datasets

CSV datasets from scraping Reddit:
* [science_25k.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_3/datasets/science_25k.csv): Dataset containing 25,000 submissions from `r/Science` 
* [philo_25k.csv.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_3/datasets/philo_25k.csv): Dataset containing 25,000 submissions from `r/Philosophy` 
* [combined_clean.csv](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_3/datasets/combined_clean.csv): Cleaned and processed datasets from `r/Science` and `r/Philosophy` subreddits

### Data Dictionary

| Feature | Type | Dataset | Description |
|---|---|---|---|
| subreddit | str | science_25k, philo_25k, combined_clean | Which subreddit each submission was scraped |
| title | str | science_25k, philo_25k | Raw strings of each submission |
| clean_text | str | combined_clean | Preprocessed strings of each submission |

------------------------------------------------
## Executive Summary
The project is split up into three code notebooks: 
1) [Scraping Subreddits](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_3/code/01_Scraping_Subreddits.ipynb): Webscraping for our datasets from `r/Science` and `r/Philosophy` using Pushshift API.
2) [Cleaning,Processing and EDA](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_3/code/02_Cleaning_Preprocessing_EDA.ipynb): Getting ready our datasets before modelling by cleaning the data, run the datasets through preprocess steps and Exploratory Data Analysis.
3) [Modelling and Conclusion](https://git.generalassemb.ly/wafirhidayat/projects/blob/master/project_3/code/03_Modelling_Conclusion.ipynb): Creating and comparing models to best choose the final estimator and concluding our project.

The project overall segments are as stated below:

### 1. Scraping Subreddits
About 25,000 submissions were scraped from each subreddits; `r/Science` and `r/Philosophy`, totaling to about 50,000 before going through cleaning process. Pushshift API was used to collect these information. Each request only give us 100 posts and to prevent being locked out by their anti-bot systems, time delay was used for each request pull and each dataframe pull per request was combined together to give us the final dataset per subreddit.

### 2. Data Cleaning and Preprocessing
The raw data is checked for any missing values for data cleaning. Following which, the raw strings in the `'title'` column for each submission was put through a series of preprocessing steps:
* Cleaning text by removing html tags (if present), links, Reddit usernames, Non-alphanumeric & new-line characters and non-english characters
* Converting each word to lower case
* Filtering out stopwords
* Stemming each word to its base form

### 3. Exploratory Data Analysis 
After cleaning and preprocessing our datasets, we carry out an initial exploratory analysis to identify some information markers from the data by various plots and graphs to visualize the features better separately. The dataframes were then combined before being put through the modelling.

### 4. Modelling
A baseline model in the form of `Multinomial Naive Bayes` with `CountVectorizer` was used and a `RandomForest Classifier` was generated as well for comparisons. Instead of creating different models one by one, [Pycaret](https://pycaret.gitbook.io/docs/) was used which is essentially a wrapper around several machine learning libraries and frameworks, such as in scikit-learn. This machine learning library will help generate different classifier models (in this case) and allows us to compare the scores in a table format for easier reference and evaluation. From there, we narrowed down to `Logistic Regression, with TFIDF Vectorizer`, as our best model as it has the best scores across our focused scoring metrics. 

### 5. Model Improvement
[Stacking Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) was used to help improve the final model more. It allows us to stack the output of individual estimator and use a final classifier to compute the final prediction. Stacking allows to use the strength of each individual estimator by using their output as input of a final estimator. 

Estimators used:
* `Multinomial Naive Bayers`
* `K Neighbors Classifier`
* `Bagging Classifier`
* `Decision Tree Classifier`
* `Extra Trees Classifier`
* `Gradient Boosting Classifier`
* `Ada Boost Classifier`

Final estimator:
* `Logistic Regression`

### 6. Evaluation
Interpretations of the confusion matrix and ROC curve for each models for comparison.

### 7. Conclusion
Concluding the project on the findings and recommendations are provided to key stakeholders involved in this project.

------------------------------------------------
## Conclusion and Recommendations

Going back to our problem statement in the beginning, we have successfully created an effective classification model using NLP techniques. Our final model incorporating the `TFIDF Vectorizer` and `Logistic Regression` (after using `Stacking Classifier`) delivered great results in terms of classifying posts correctly, indicating high predictive performance. The model was also good for inference, as we were able to extract the feature importances from the model from its coefficient. Although it would not be able to 100% accurately classify submissions to their correct subreddit, it will be able to give a gauge on the majority of the submissions based on our dataset.

One limitation is that our model is able to accurately classify these submission between `r/Science` or `r/Philosophy` is because the latter is plagued with spam posts while the former barely has any. This could be a potential reason for its high success rate as a post with high spam words will be classified in `r/Philosophy` which defeats the purpose of our project. 
An improvement to this model would be to gather more data in a larger timeframe and thus having a larger pool of information. Even when we clean the datasets of bots submission and spam, there will still be sufficient data if the data was vast in the first place. 

In addition, the timespan of posts is also constrained and the data may thus be affected by certain events that do not consistently happen. For instance, our EDA showed that COVID-19 were prevalent in both subreddits. However, this was definitely not the case before the year 2020, and perhaps our model would perform very differently if given the subreddit data for a much longer time frame. 

We could also try and incorporate more semantic concepts into our model such as a measure of sentiment analysis to help add another feature to classify posts. For example using a pre-trained model from [HuggingFace](https://blog.tensorflow.org/2019/11/hugging-face-state-of-art-natural.html) which has many sentiment analysis trained models. All of this can be accumulated to making an app to show where redditors can post their intended submissions and show users the sentiment analysis of their text (for example the submitted posts are positive/negative and subjective/objective in nature) which would be more useful for redditors and subreddit moderators.

