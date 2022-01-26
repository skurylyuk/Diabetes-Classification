Diabetes-Classification
# Data 

# Features 
* High BP (0 = no high BP, 1 = high BP)
* High Cholesterol (0 = no high Cholesterol, 1 = high Cholesterol)
* Cholesterol Check (0 = no cholesterol check in 5 years,  1 = yes cholesterol check in 5 years)
* Body Mass Index
* Smoker 
* Stroke (0 = no,  1 = yes)
* Heart Disease or Attack (Coronary heart disease (CHD) or myocardial infarction (MI) 0 = no,  1 = yes)
* Physical Activity (physical activity in past 30 days - not including job 0 = no, 1 = yes)
* Fruits (Consume Fruit 1 or more times per day 0 = no, 1 = yes)
* Veggies (Consume Vegetables 1 or more times per day 0 = no, 1 = yes)
* Heavy Alcohol Consumption (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week, 0 = no)
* Any Healthcare (Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no, 1 = yes)
* No Doctor because of cost (Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no, 1 = yes)
* General Health (Would you say that in general your health is: scale 1-5 - 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)
* Mental Health 
* * Physical Health
* Difficulty Walking (Do you have serious difficulty walking or climbing stairs? 0 = no, 1 = yes)
* Sex (0 = female, 1 = male)
* Age (13-level age category)
  * 1 = Age 18 - 24
  * 2	= Age 25 to 29
  * 3	= Age 30 to 34
  * 4	= Age 35 to 39
  * 5	= Age 40 to 44
  * 6	= Age 45 to 49
  * 7	= Age 50 to 54
  * 8	= Age 55 to 59
  * 9	= Age 60 to 64
  * 10 = Age 65 to 69
  * 11 = Age 70 to 74
  * 12 = Age 75 to 79
  * 13 = Age 80 or older
  
* Education
  * 1 =	Never attended school or only kindergarten
  * 2	= Grades 1 - 8 (Elementary)
  * 3	= Grades 9 - 11 (Some high school)
  * 4	= Grade 12 or GED (High school graduate)
  * 5	= College 1 year to 3 years (Some college or technical school)
  * 6	= College 4 years or more (College graduate)
* Income
  * 1 = Less than $10,000
  * 2 = $10,000 to less than $15,000
  * 3 = $15,000 to less than $20,000
  * 4 = $20,000 to less than $25,000
  * 5 = $25,000 to less than $35,000
  * 6 = $35,000 to less than $50,000
  * 7 = $50,000 to less than $75,000
  * 8 = $75,000 or more
# Logistic Regression: 
* Accuracy: 0.7896 
* Precision: 0.3639 
* Recall: 0.6476 
* F1: 0.4659
* Cross Validation Accuracy Score : 0.74455

# 10 nearest neighbors validation metrics: 
* Accuracy: 0.7673 
* Precision: 0.3128 
* Recall: 0.5360 
* F1: 0.3951
* Cross Validation Accuracy Score : 0.77854

# Random Forest validation metrics: 
* Accuracy: 0.7698 
* Precision: 0.3488 
* Recall: 0.7195 
* F1: 0.4698
* Cross Validation Accuracy Score: 0.80256

# Decision Tree validation metrics: 
* Accuracy: 0.7699 
* Precision: 0.3330 
* Recall: 0.6218 
* F1: 0.4337
* Cross Validation Accuracy Score: 0.71748

# Stacking Ensembling validation metrics: 
* Accuracy: 0.8623 
* Precision: 0.5388 
* Recall: 0.1988 
* F1: 0.2905
* Cross Validation Accuracy Score: 0.92498

# Voting Ensembling validation metrics: 
* Accuracy: 0.7974 
* Precision: 0.3560 
* Recall: 0.5296
* F1: 0.4258
* Cross Validation Accuracy Score: 0.8623
