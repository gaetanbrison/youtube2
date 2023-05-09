import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from six import StringIO  
import pydotplus

df = pd.read_csv("youtube.csv",delimiter=";")
app_mode = st.sidebar.selectbox('Select Page',['Summary 🚀','Visualization 📊','Prediction 📈'])

# Read HTML file
#with open('example_report.html', 'r') as f:
#    html_string = f.read()

# Display HTML file
#st.markdown(html_string, unsafe_allow_html=True)
### The st.title function sets the title of the web application to "Final Project - 01 Introduction Page".
if app_mode == 'Summary 🚀':
    st.title("Final Project - 01 Introduction Page")

    ### The first two lines of the code load an image and display it using the st.image function.
    image_logo = Image.open('youtube.png')
    st.image(image_logo, width=300)

    ### The st.subheader function sets the title of the web application to "YouTuber Data Analysis".
    st.subheader("YouTuber Data Analysis")
    st.markdown("##### Analyzing: If I want to quit my job and become a full time YouTuber, what category of YouTube channel should I start and where? The definition of YouTuber success is defined by the number of subscribers")


    ### The st.number_input function creates a widget that allows the user to input a number. The st.radio function creates a radio button widget that allows the user to select either "Head" or "Tail".
    num = st.number_input('No. of Rows', 5, 100)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
    ### the st.dataframe function displays the data frame.
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    ### The st.markdown function is used to display some text and headings on the web application.
    st.markdown("### 01 - Show  Dataset")

    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    ### The st.text and st.write functions display the shape of the data frame and some information about the variables in the data set.
    st.text('(Rows,Columns)')
    st.write(df.shape)

    st.markdown("##### variables ➡️")
    st.markdown(" **Username**: qualitative identification for each YouTuber")
    st.markdown(" **Name**: screen name for each YouTuber")
    st.markdown(" **Category**: qualitative categorization for each YouTuber")
    st.markdown(" **Subscribers**: number amount of subscribers for each YouTuber")
    st.markdown(" **Audience Country**: localization of the majority of viewers for each YouTuber")
    st.markdown(" **Avg Views**: quantitative data for average views for each YouTuber")
    st.markdown(" **Avg Likes**: quantitative data for average likes for each YouTuber")


    st.dataframe(df.head(3))


    ### The st.markdown and st.dataframe functions display the descriptive statistics of the data set.
    st.markdown("### 02 - Description")
    st.dataframe(df.describe())


    ### The st.markdown, st.write, and st.warning functions are used to display information about the missing values in the data set.
    st.markdown("### 03 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")


    st.markdown("### 04 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")    
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")
















if app_mode == 'Visualization 📊':
    st.subheader("02 Visualization Page - Youtube Data Analysis 📊")
    #response = requests.get("https://lookerstudio.google.com/u/0/reporting/97c91d4e-e116-488f-b695-50179f0c7a11/page/pYAKD")
    #html_code = response.text
    #st.write("My Looker Dashboard")
    #html_code = f'<iframe srcdoc="{html_code}" width="100%" height="600" frameborder="0"></iframe>'
    #html(html_code)

    st.markdown("[![Foo](https://i.postimg.cc/9FWpBqw7/Screenshot-2023-05-09-at-14-59-09.png)](https://lookerstudio.google.com/reporting/7b719045-7fa5-4209-b3fd-161cd2483764)")

    #image_dashboard = Image.open('images/dashboard.png')
    #st.image(image_dashboard)


if app_mode == 'Prediction 📈':


    # Load the dataset
    df = pd.read_csv('youtube2.csv')

    # Factorize
    df['Category'] = pd.factorize(df['Category'])[0]
    df['Audience Country'] = pd.factorize(df['Audience Country'])[0]
    df['username'] = pd.factorize(df['username'])[0]
    df['Name'] = pd.factorize(df['Name'])[0]


    # Define the model selection dropdown
    model_type = st.selectbox('Select a model to use', ['KNN', 'Decision Tree'])

    ### The st.title() function sets the title of the Streamlit application to "Final Project - 03 Prediction Page".
    st.title("Final Project - 03 Prediction Page")

    if model_type == 'KNN':
        # Split the dataset into features and target
        X = df.drop('Category', axis=1)
        y = df['Category']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create the kNN classifier
        k = 5
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the kNN classifier
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(X_test)

        # Evaluate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        # The st.columns() function creates two columns
        col1,col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(X.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))

        # The st.subheader() function creates a subheading for the results section.
        st.subheader('🎯 Results')

        # The st.write() function displays differnt metrics for the kNN model, including 
        st.write("1) The model explains", confusion_matrix(y_test, y_pred),"% variance of the target feature")
        st.write("2) Here is the classification report:", metrics.classification_report(y_test,y_pred))
        st.write("3) Accuracy:", accuracy)

    elif model_type == 'Decision Tree':
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df.drop('Subscribers', axis=1), df['Subscribers'],test_size=0.1,random_state=42)
            
        # Train the decision tree model
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(X_train, y_train)

        # Make predictions on the testing set
        predictions = decision_tree_model.predict(X_test)

        # Evaluate the accuracy of the model
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy:.2f}')

        # The st.subheader() function creates a subheading for the results section.
        st.subheader('🎯 Results')

        # The st.write() function displays differnt metrics for the kNN model, including 
        st.write("1) The model explains", confusion_matrix(y_test, predictions),"% variance of the target feature")
        st.text("2) Here is the classification report:", metrics.classification_report(y_test,predictions))
        st.write("3) Accuracy", accuracy)


        ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
        list_variables = df.columns
        select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

        ### The st.sidebar.number_input() function creates a number input widget in the sidebar that allows users to select the size of the training set.
        train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)

        new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
