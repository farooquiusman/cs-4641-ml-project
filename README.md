# House Price Prediction Model
Project Group 114: Nick Bergeron, Taeho Park, Usman Farooqui, Shubham Jha

## Introduction/Background
This proposal describes a Machine Learning project that uses the dataset from Zillow to forecast home values. Our goal is to create a prediction model that uses a variety of features and past data to reliably estimate residential property values. The House Price Prediction Model project addresses the growing demand for reliable property valuation models, benefiting homebuyers, sellers, and investors. We will use machine learning methods to build a strong model for high-accuracy property value estimation by utilizing a diversified dataset that includes property attributes, geographical data, and historical pricing information [1].  This proposal will go into detail about our technique, approach, and possible effects of our projections on the housing market.

## Problem Definition
Purchasing a home is a major financial investment for countless people in the US, and with market fluctuations, it can be hard to know the optimal time to buy a home. With this predictive model, potential homeowners can gauge a timeline for the ideal opportunity to purchase a home, providing some peace of mind and clarity. Additionally, this model can indicate the ideal time for homeowners to sell by predicting future house values. Overall, our model will help bring some certainty to an otherwise volatile market.

## Methods
### Supervised Learning

#### Random Forest Regressor:
This approach predicts continuous numerical values and captures intricate and nonlinear associations between house prices and features. A mix of  parametric and non-parametric regression provides the most accurate models [2].

#### Linear Regressor:
Linear regression is valuable for modeling the connection between a dependent variable (the target or output) and multiple independent variables (predictors or features). This is particularly advantageous predicting SalePrice based on features like Neighor, YearRenovated, and others.

#### Libraries:
Pandas, Matplotlib, Seaborn

## Potential Results and Discussion
To determine the accuracy of our model we will first train the model using older housing data and compare the expected value to the actual value using resources such as Zillow. They used a similar process to check the validity of their model [3]. Once our model’s accuracy has been verified across several geographic areas and parameters we will be able to use it to predict future housing trends and pricing. Ensuring our model works across multiple parameters is very important, as there are a multitude of factors such as: weather, crime, schools, and income that affect housing prices and our model needs to take all of them into account. For our model to be truly universal it needs to be able to be accurate over a wide variety of geographical areas. Our acceptable margins for error will be decided later once we are at a point to begin testing and comparing results.
## Results and Discussion
### Dataset Breakdown
Our dataset consists of property values [provided by Kaggle]([url](https://www.kaggle.com/datasets/camnugent/california-housing-prices)) from the 1990 California census for all residential properties in each district. This dataset was very straightforward to work with and encompassed more than enough datapoints for our training model to work with while giving a wide variety of types of properties for our model. For each property, there are varables describing information about both the house and the people living in it. Each row corresponds to a census block group, which is the smalles geographical unit the U.S. census publishes data for. While the data may seem old, age does not matter for the purpose of training our model, and inflation is very easy to adjust for by year. 
## Gantt Chart

## Gantt Chart
https://bit.ly/ml-gantt-chart

## References
[1] N. Ghosalkar and S. Dhage, Real estate value prediction using linear regression | IEEE conference ..., https://ieeexplore.ieee.org/document/8697639 (accessed Oct. 6, 2023).\
[2] R. Gencay et al., “A prediction comparison of housing sales prices by parametric versus semi-parametric regressions,” Journal of Housing Economics, https://www.sciencedirect.com/science/article/abs/pii/S105113770400004X (accessed Oct. 6, 2023).\
[3] A. Nguyen, Housing price prediction,https://cs.union.edu/Archives/SeniorProjects/2018/CS.2018/files/nguyena2/nguyena2-499-report.pdf (accessed Oct. 6, 2023).
