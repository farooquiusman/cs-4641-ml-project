<img width="451" alt="Screenshot 2023-11-05 at 7 57 52 PM" src="https://github.com/farooquiusman/house-price-prediction-model/assets/113317744/4d2bddf8-2379-48e7-b2d3-f66799513a02">

# House Price Prediction Model
Project Group 114: Nick Bergeron, Taeho Park, Usman Farooqui, Shubham Jha


## Introduction/Background
This proposal describes a Machine Learning project that uses the dataset from Zillow to forecast home values. Our goal is to create a prediction model that uses a variety of features and past data to reliably estimate residential property values. The House Price Prediction Model project addresses the growing demand for reliable property valuation models, benefiting homebuyers, sellers, and investors. We will use machine learning methods to build a strong model for high-accuracy property value estimation by utilizing a diversified dataset that includes property attributes, geographical data, and historical pricing information [1].  This proposal will go into detail about our technique, approach, and possible effects of our projections on the housing market.

## Problem Definition
Purchasing a home is a major financial investment for countless people in the US, and with market fluctuations, it can be hard to know the optimal time to buy a home. With this predictive model, potential homeowners can gauge a timeline for the ideal opportunity to purchase a home, providing some peace of mind and clarity. Additionally, this model can indicate the ideal time for homeowners to sell by predicting future house values. Overall, our model will help bring some certainty to an otherwise volatile market.

## Methods
### Supervised Learning

#### Data Preprocessing:
<img width="831" alt="Screenshot 2023-11-05 at 8 21 37 PM" src="assets/css/histogram_image.png"> (histograms)

During the initial phase of our data cleaning process, we identified and eliminated all instances of null entries and any data points that were deemed redundant or of no value to our analysis. Our examination of the dataset through histograms revealed that the distribution of variables such as total_rooms, total_bedrooms, population, and households is notably right-skewed. To address this and normalize the data we applyied a logarithmic transformation , which brought the distribution of these variables closer to a standard bell curve shape.

Furthermore, we explored the relationship between ocean proximity and housing prices. The visual data suggests that properties located nearer to the coast tend to have higher median house values. To quantitatively assess this observation, we have implemented one-hot encoding for the 'ocean_proximity' feature. This transformation will enable us to incorporate this categorical variable into our models more effectively and examine its potential impact on the median house value.

Lastly, we created a new heatmap to visually represent the correlations between housing value and the newly encoded 'ocean_proximity' data, along with other relevant variables. This will help in gaining a more nuanced understanding of the factors that influence house prices in our dataset.

#### Linear Regressor:
we developed a linear regression model using Python's scikit-learn library to predict median house values based on a set of housing-related attributes. After importing the necessary modules, we prepared our training data by dropping the target variable 'median_house_value' from our feature set and assigning it to our target variable, y_train. We then instantiated the LinearRegression model and fitted it to our training data to create a predictive model. To ensure the model's compatibility with the test data, we also performed similar preprocessing steps on the test set, including joining features and target variables, transforming certain features using the natural logarithm for normalization purposes, and handling categorical data. Finally, we evaluated the model's accuracy by predicting the median house values of the test set and calculating metrics such as the mean squared error and the coefficient of determination, or R² score.

#### Libraries:
Pandas, Matplotlib, Seaborn

## Potential Results and Discussion
To determine the accuracy of our model we will first train the model using older housing data and compare the expected value to the actual value using resources such as Zillow. They used a similar process to check the validity of their model [3]. Once our model’s accuracy has been verified across several geographic areas and parameters we will be able to use it to predict future housing trends and pricing. Ensuring our model works across multiple parameters is very important, as there are a multitude of factors such as: weather, crime, schools, and income that affect housing prices and our model needs to take all of them into account. For our model to be truly universal it needs to be able to be accurate over a wide variety of geographical areas. Our acceptable margins for error will be decided later once we are at a point to begin testing and comparing results.
## Results and Discussion
### Dataset Breakdown
Our dataset consists of property values [provided by Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) from the 1990 California census for all residential properties in each district. This dataset was very straightforward to work with and encompassed more than enough datapoints for our training model to work with while giving a wide variety of types of properties for our model. For each property, there are varables describing information about both the house and the people living in it. Each row corresponds to a census block group, which is the smalles geographical unit the U.S. census publishes data for. While the data may seem old, age does not matter for the purpose of training our model, and inflation is very easy to adjust for by year. 
### Results
After training our linear regression model, the next step was to analyze our accuracy using both Root Mean Squared Error and a score for the Coefficient of Determination Regression. For our linear regression model, the RMSE and Coefficient of Determination Regression was 65150.27 and 0.6793 respectively. The RMSE is not normalized, so on average our model was off by around $65000 for each property value. As expected a linear regression model is not the most accurate at predicting the real data points, but the primary reason of the RMSE and Coefficient of Determination Regression is to compare the accuracy for each of our models. Once we have more working models to compare these results to, we can begin determing acceptable margins of error for each model. Overall our model had about an average performance, the dataset was fairly straightforward and easy to work with which helped, but there is room to improve with this model. Primarily in caluclating the influence each variable has on the overall property value. Beyond just analyzing accuracy, our results also included visualizing the effect of ocean proximity on housing prices to show how one variable can have a significant impact on the property value. By graphing each data point using it's latitude and longitude, it clearly shows how proximity to the ocean can raise the property value: 
![Geographical Property Values](/house-price-prediction-model/assets/css/propertyvalue_coordinates.png)
Along with this geographical visualization, a heatmap was created using one-hot encoding for the "ocean_proximity features".
![image](https://github.com/farooquiusman/house-price-prediction-model/assets/122696234/30722734-a14e-4492-aa2c-328a2c156a9d)
This visualization will allow us to better understand how each variable affects the overall property value, and improve our model going forward.


## Next Step
Our next step involves implementing a Random Forest Regressor, a reliable and straightforward method known for producing accurate results. Here's a step-by-step guide:
1. Data Collection and Preprocessing: Begin by gathering data encompassing various house features, such as square footage, the number of bedrooms, location, and more. Preprocess the data to address missing values, encode categorical variables, and if necessary, normalize or scale numerical features.
2. Data Splitting: Divide your dataset into two essential parts: a training set and a test set. This division is crucial for assessing the model's performance.
3. Random Forest Regressor Creation and Training: Initialize a Random Forest Regressor, carefully setting its hyperparameters. Then, train the model using the training data.
4. Prediction Generation: Utilize the trained model to make predictions on the test data, allowing you to evaluate its predictive capabilities.
5. Model Evaluation: Assess the model's performance using common regression metrics like Root Mean Squared Error (RMSE). Lower values of these metrics indicate more accurate predictions. Consider additional evaluation metrics like R-squared (R^2) to further gauge the model's explanatory power.
6. Model Refinement: As you progress, keep in mind that not all features are equally significant. Continually assess the importance of each feature and consider feature selection or engineering to enhance the model's performance. Fine-tune hyperparameters as needed and retrain the model to achieve optimal results.

This approach ensures a systematic and effective process while emphasizing the importance of feature selection and hyperparameter tuning for a successful Random Forest Regressor implementation.

## Gantt Chart
https://bit.ly/ml-gantt-chart

## References
[1] N. Ghosalkar and S. Dhage, Real estate value prediction using linear regression | IEEE conference ..., https://ieeexplore.ieee.org/document/8697639 (accessed Oct. 6, 2023).\
[2] R. Gencay et al., “A prediction comparison of housing sales prices by parametric versus semi-parametric regressions,” Journal of Housing Economics, https://www.sciencedirect.com/science/article/abs/pii/S105113770400004X (accessed Oct. 6, 2023).\
[3] A. Nguyen, Housing price prediction,https://cs.union.edu/Archives/SeniorProjects/2018/CS.2018/files/nguyena2/nguyena2-499-report.pdf (accessed Oct. 6, 2023).
