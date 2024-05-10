import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#LOAD DATA
df = pd.read_csv('economyindicators.csv')

#SIDEBAR
def main():
    # Menggunakan CSS untuk mengatur warna sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #164863;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.title("Global Economy Growth")
    dropdown_options = ["Business Understanding", "Exploratory Data Analysis", "Evaluation"]
    selected_dropdown = st.sidebar.selectbox("Navigation", dropdown_options)
    
    #MAIN CONTENT
    #BU
    if selected_dropdown == "Business Understanding":
        # Menggunakan CSS untuk mengatur warna halaman utama
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #9BBEC8; 
                color: black; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write(" # Global Economy Growth")
        st.caption("The provided dataset is a collection of global economic indicators. It encompasses a wide range of economic information from various countries worldwide, including data on economic growth, international trade, investment, and more. With this dataset, users can analyze global economic trends, identify factors influencing economic growth, and gain valuable insights for economic decision-making.")
        st.write("#### Business Understanding")
        st.write("##### Business Objective")
        st.caption("The rationale behind this research on Global Economy Growth is to gain insights into the factors influencing economic development worldwide. By analyzing a comprehensive dataset, the aim is to uncover patterns, trends, and relationships that contribute to economic growth across different countries and regions. Understanding these dynamics is essential for policymakers, economists, and businesses to make informed decisions and implement strategies that foster sustainable economic progress on a global scale.")
        st.write("##### Assess Situation")
        st.caption("The situation that results from the complex interplay of various economic indicators is multifaceted. Global economies experience fluctuations driven by factors such as GDP growth rates, population dynamics, trade balances, investment patterns, and government policies. These variables interact in intricate ways, impacting the overall economic landscape and influencing decision-making processes. Additionally, external factors such as geopolitical tensions, natural disasters, and technological advancements further complicate the assessment of the global economic situation.")
        st.write("##### Data Mining Goals")
        st.caption("The ultimate objective of this data mining endeavor is to extract meaningful insights from the Global Economy Indicators dataset. By applying advanced analytics techniques, including exploratory data analysis, regression analysis, clustering, and time series forecasting, the goal is to uncover hidden patterns, identify key drivers of economic growth, and predict future trends. Through this process, we aim to enhance our understanding of the complex dynamics shaping the global economy and provide valuable insights for policymakers, researchers, and businesses to support evidence-based decision-making.")
        st.write("##### Project Plan")
        st.caption("The project encompasses several key steps to achieve its objectives. Firstly, data preprocessing will be conducted to clean and prepare the 'economyindicators.csv' dataset for analysis, including handling missing values, standardizing data formats, and encoding categorical variables. Next, exploratory data analysis will be performed to gain an initial understanding of the dataset's structure, distribution, and relationships between variables. Subsequently, advanced analytics techniques will be applied to uncover patterns and relationships within the data, including regression analysis to model the impact of various factors on economic growth and clustering to identify distinct groups of countries based on economic indicators. Finally, the findings will be interpreted and communicated through insightful visualizations and comprehensive reports, providing valuable insights into the dynamics of global economic growth.")

    #EDA
    elif selected_dropdown == "Exploratory Data Analysis":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #9BBEC8; 
                color: black; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write("# Exploratory Data Analysis")
        st.caption("In the Exploratory Data Analysis (EDA) phase, we embark on a journey to gain insights and understanding from the Global Economy Indicators dataset. This crucial step involves examining various economic metrics and visualizing their patterns and relationships to uncover valuable insights into the dynamics of the global economy.")

        st.write("#### Population Growth")
        # Line chart for Population
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x='Year', y='Population', ax=ax)
        plt.title('Population Growth Over Time')
        plt.xlabel('Year')
        plt.ylabel('Population')
        st.pyplot(fig)
        st.caption("Population plays a crucial role as it represents the total number of individuals living within a particular region or country. The population growth over time, depicted in the line chart, illustrates the changing dynamics of a nation's demographic landscape. Understanding population trends is essential for policymakers, economists, and businesses as it directly influences various aspects of the economy, including labor force availability, consumer demand, and resource utilization.")
        st.write("###### Interpretation and Actionable Insights:")
        st.caption("The visualization of population growth over time reveals significant insights into demographic trends. Rapid population growth may indicate potential opportunities for market expansion and increased consumer demand. Policymakers could use this information to plan infrastructure development, healthcare services, and educational programs to accommodate population growth and support sustainable economic development.")

        st.write("#### Gross Domestic Product Growth")
        # Line chart for GDP
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x='Year', y='Gross Domestic Product (GDP)', ax=ax)
        plt.title('GDP Growth Over Time')
        plt.xlabel('Year')
        plt.ylabel('GDP')
        st.pyplot(fig)
        st.caption("GDP growth is a key determinant of a country's economic health and development. A rising GDP indicates expanding economic activity, increased productivity, and higher living standards for the population. It signifies growing business opportunities, job creation, and investment potential, fostering prosperity and socio-economic advancement.")

        st.write("#### Distribution of Per Capita GNI")
        # Histogram for Per Capita GNI
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['Per capita GNI'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Per Capita GNI')
        ax.set_xlabel('Per Capita GNI')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        st.caption("The histogram above illustrates the distribution of Per Capita Gross National Income (GNI) across the dataset. This visualization provides insights into the frequency of different ranges of Per Capita GNI values within the economy. Understanding the distribution of Per Capita GNI is crucial for assessing the economic well-being of individuals within a country or region. Higher values indicate greater average income levels, which can contribute to overall economic growth and development. Analyzing the distribution of Per Capita GNI helps policymakers, economists, and businesses understand income disparities, formulate targeted policies, and identify opportunities for socio-economic advancement.")

        st.write("#### AMA Exchange Rate")
        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        df['AMA exchange rate'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black', ax=ax)
        plt.title('AMA Exchange Rate')
        plt.xlabel('AMA Exchange Rate')
        plt.ylabel('Frequency')
        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Show histogram in Streamlit
        st.pyplot(fig)
        st.caption("The AMA exchange rate refers to the exchange rate provided by the Amadeus Market Analytics (AMA) service. This exchange rate represents the conversion rate between two currencies, typically denoting the value of one currency in terms of another. In the context of global economy indicators, the AMA exchange rate signifies the relative value of currencies in international markets, reflecting the dynamics of foreign exchange markets.")

        st.write("#### AMA Exchange Rate vs IMF Based Exchange Rate")
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='scatter', x='AMA exchange rate', y='IMF based exchange rate', s=32, alpha=.8, ax=ax)
        plt.title('AMA Exchange Rate vs IMF Based Exchange Rate')
        plt.xlabel('AMA Exchange Rate')
        plt.ylabel('IMF Based Exchange Rate')
        # Hide spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Show scatter plot in Streamlit
        st.pyplot(fig)
        st.caption("The comparison between AMA exchange rate and IMF-based exchange rate provides valuable insights into the discrepancies or agreements between two different sources of exchange rate data: the Amadeus Market Analytics (AMA) service and the International Monetary Fund (IMF). The AMA exchange rate represents the exchange rate provided by the Amadeus Market Analytics service, which may utilize proprietary methodologies or data sources to calculate exchange rates. On the other hand, the IMF-based exchange rate refers to exchange rate data sourced from the International Monetary Fund, an organization that collects and analyzes economic data from member countries worldwide.")

    elif selected_dropdown == "Evaluation":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #9BBEC8; 
                color: black; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write("# Evaluation")
        st.caption("During the evaluation phase, we delve into assessing the performance of our predictive models. This critical step involves a comprehensive analysis using multiple metrics to gauge how well our models are performing in predicting the target variable. The evaluation metrics employed in this phase provide insights into different aspects of model performance, aiding in the interpretation and comparison of model effectiveness.")

        # Fungsi untuk menghitung MSE
        def mean_squared_error(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        # Fungsi untuk menghitung MAE
        def mean_absolute_error(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        # Fungsi untuk menghitung R-squared (R2)
        def r2_score(y_true, y_pred):
            mean_y_true = np.mean(y_true)
            total_sum_squares = np.sum((y_true - mean_y_true) ** 2)
            residual_sum_squares = np.sum((y_true - y_pred) ** 2)
            r2 = 1 - (residual_sum_squares / total_sum_squares)
            return r2

        # Placeholder data
        models = ['KNN Regression', 'DTR Regression']
        mse_scores = [29.1191592728361, 46.010323639181216]
        mae_scores = [1.462965278918901, 0.6880234769312534]
        r2_scores = [0.9884517736701494, 0.9817529886108147]

        # Evaluasi KNN Regression (contoh placeholder data)
        mse_knn, mae_knn, r2_knn = 29.1191592728361, 1.462965278918901, 0.9884517736701494

        # Evaluasi DTR Regression (contoh placeholder data)
        mse_dtr, mae_dtr, r2_dtr = 46.010323639181216, 0.6880234769312534, 0.9817529886108147

        st.write("#### Mean Squared Error (MSE)")
        # Plotting MSE
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_chart = ax.bar(models, mse_scores, color='skyblue')
        ax.set_title('Mean Squared Error (MSE) Comparison')
        ax.set_xlabel('Regression Models')
        ax.set_ylabel('MSE')
        ax.set_ylim(0, max(mse_scores) * 1.1)
        # Convert matplotlib figure to streamlit
        st.pyplot(fig)
        st.caption("MSE provides insight into how well a regression model is making accurate predictions. A smaller MSE value indicates that the model is better at fitting the data, as it suggests that the model's predicted values tend to be closer to the actual values of the observed data. Conversely, if the MSE value is large, it indicates that the model tends to provide predictions that are far from the actual values.")

        st.write("#### Mean Absolute Error (MAE)")
        # Plotting MAE
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_chart = ax.bar(models, mae_scores, color='lightgreen')
        ax.set_title('Mean Absolute Error (MAE) Comparison')
        ax.set_xlabel('Regression Models')
        ax.set_ylabel('MAE')
        ax.set_ylim(0, max(mae_scores) * 1.1)
        # Convert matplotlib figure to streamlit
        st.pyplot(fig)
        st.caption("MAE provides a straightforward interpretation: it represents the average absolute deviation between the predicted and actual values. Unlike MSE, which squares the errors, MAE gives equal weight to all errors, making it less sensitive to outliers. A lower MAE indicates better model performance, with predictions that are, on average, closer to the actual values.")

        st.write("#### R-Squared (R2)")
        # Plotting R2 Score
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_chart = ax.bar(models, r2_scores, color='salmon')
        ax.set_title('R-squared (R2) Score Comparison')
        ax.set_xlabel('Regression Models')
        ax.set_ylabel('R2 Score')
        ax.set_ylim(0, 1.1)
        # Convert matplotlib figure to streamlit
        st.pyplot(fig)
        st.caption("R-squared (R2) is a statistical measure that represents the proportion of variance in the dependent variable that is explained by the independent variables in a regression model. It is also known as the coefficient of determination. A higher R2 value indicates a better fit, meaning that the independent variables explain a larger proportion of the variance in the dependent variable.")




main()