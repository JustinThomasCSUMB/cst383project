**Intro**

Our group decided to use a data set from Kaggle which is meant to show correlation between a patients health (things such as BMI, smoking, age, etc…) and their likelihood of having a stroke.  The objective of this project was to be able to take the data set and determine the probability of a specific person having a stroke based on their current health factors. While the data set provided a lot of patient information we hypothesized that not all of the information presented was pertinent in predicting the likelihood of a patient having a stroke. From this research it would help to eliminate the amount of data needed to be processed when assessing the patients risk of having a stroke, thus leading to a faster, and potentially earlier, response.


**Selection of Data**

The data set we used has a total of 5,110 entries, each with 12 columns. The columns are the patient’s ID, gender, age, hypertension, heart disease, marital status, work type, living conditions, glucose level, and body mass index. Of the 5,110 entries only 201 rows were found to have null values – about 4%. The rows that were found to have null values were dropped since it would have very little impact on the integrity of the data.

Of the twelve columns we chose to remove marital status, work type, living conditions, and patient ID as we believed them to not be relevant in exploring our hypothesis.

Of the remaining columns there is one that is a string: Gender, one that is an integer: Age, two that are floats:  glucose level, body mass index, lastly there are three which are boolean values stored as binary: hypertension, heart disease, and smoking status.

We used data munching to convert the Gender category to match the other binary columns. This allowed us to process the data much more efficiently. Thus leading to a faster turn around time when testing our data.

**Methods**

Tools: 
  1. Numpy, Padnas, Mathplotlib, and Seaborn for data analysis and visualization
  2. Scikit-learn for inference
  3. Github for data management
  4. Anaconda for Jupiter Notebook

Inference methods used with Scikit:
  1. Logistic Regression
  2. Standard Scaling
  3. Roc Curve
  4. Confusion Matrix
  5. Precision Recall Curve
  6. R2 Score
  7. Train Test Split

**Results**

![Graph](https://i.imgur.com/cIebX23.png)

 When we tested all columns our results showed that our test results were accurate until around 96% where they then start to plateau as the results get flooded with false positives. From the confusion matrix we are able to determine that the vast majority of our results are true positives with only 54 false negatives and one true negative. 

The second set of graphs shows the results when we test only the columns that our team deemed necessary. The graphs from testing all columns and the graphs from testing our selected columns are nearly identical, with the confusion matrix being an exact copy. Thus giving our hypothesis validity.

The final set of graphs are the columns we decided to cut. These columns are the largest source of variance, however the variance is so minor that when tested alongside the other columns it barley affects the results. Since this is the case it confirms our hypothesis that the columns were unneeded and were simply bloating the data.

**Discussion**

From our results we were able to extrapolate that from our selected data columns we were able create results almost identical to the results produced when testing all of the data. Of all the data we used 70% of it was dedicated to training, 30% for testing, and overall the test accuracy was 96% in all cases. Our results seem to prove our hypothesis that categories such as marital status, work type, and living conditions are almost irrelevant when determining the risk of stroke.

**Summary**

The purpose of this project was to show that certain types of data were not needed when it came to predicting the likelihood of a patient suffering from a stroke. In order to test this hypothesis we compared and contrasted the results produced when running both the full set of data and the data from the category we thought were actual contributors. Since the results had such a high level of parity we deduced that our hypothesis was correct and factors such as  marital status, work type,  and living conditions have little to no role in the probability of a patient suffering form a stroke.

Github Link: https://github.com/JustinThomasCSUMB/cst383project

Video Link: https://youtu.be/Tn6ychPENyI
