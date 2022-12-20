# Importing all the Required Libraries

import os
import streamlit as st

# EDA Packages
import pandas as pd

# Viz Packages
import matplotlib
import matplotlib.pyplot
import seaborn as sns

def main():
    """Custom ML Dataset Explorer Function"""
    st.title("Machine Learning Dataset Explorer")
    st.subheader("Custom Dataset Explorer Application using Streamlit")

    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is None:

        # Selecting a File/Dataset
        def file_selector(folder_path = './datasets'):
            filenames = os.listdir(folder_path)
            selected_file = st.selectbox("Select a file", filenames)
            return os.path.join(folder_path,selected_file)

        # Outputting the Selected File
        filename = file_selector()
        st.info("File Selected {}".format(filename))

        # Reading the Data
        df = pd.read_csv(filename)

        # Showing the Dataset
        if st.checkbox("Show Dataset"):
            number = st.number_input("Number of Rows to View",5,100)
            st.dataframe(df.head(number))

        st.subheader("Data Level Exploratory Data Analysis(EDA)")

        # Showing the Description of the Dataset
        if st.checkbox("Description of the Dataset : Some Basic Statistics"):
            st.write(df.describe().T)

        # Showing the Columns
        if st.checkbox("Column Names"):
            st.write(df.columns)
    
        # Showing DataTypes of the Columns
        if st.checkbox("Data Types"):
            st.write(df.dtypes)

        # Selective filtering of Columns to Show
        if st.checkbox("Select Columns to Show"):
            all_cols = df.columns.to_list()
            selected_cols = st.multiselect("Select Columns",all_cols)
            new_df = df[selected_cols]  
            st.dataframe(new_df)

        # Vlaue Counts per Selected Column    
        if st.checkbox("Value Counts"):
            cols = df.columns.to_list()
            col = st.selectbox("Select a Column",cols)
            st.text("The Value Counts of the Selected Columns are as follows :")
            st.write(df[col].sort_values(ascending=False).value_counts())

        # Showing the Shape of Dataset
        if st.checkbox("Shape of the Dataset"):
            data_dim = st.radio("Show Dimensions by ",("Rows","Columns"))
            if data_dim == 'Rows':
                st.text("Number of Rows")
                st.write(df.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df.shape[1])
            else:
                st.write(df.shape)

        # Showing the Null Values by Column
        if st.checkbox("Checking For Null Values"):
            st.write(df.isna().sum())

        st.subheader("Data Visualization : Visual Level EDA")

        # Count Plots

        if st.checkbox("A Value Counts Plot"):
            st.text("Value Counts by Target")
            all_col_names = df.columns.to_list()
            primary_col = st.selectbox("Primary Column to GroupBy", all_col_names)
            selected_col_names = st.multiselect("Select Columns", all_col_names)
            if st.button("Plot"):
                st.text("Generate Plot")
                if selected_col_names:
                    vc_plot = df.groupby(primary_col)[selected_col_names].count()
                else:
                    vc_plot = df.iloc[:,-1].value_counts()
                st.write(vc_plot.plot(kind="bar"))
                st.pyplot()

        # Seaborn Plots

        if st.checkbox("Correlation Plot[using Seaborn]"):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()

        # Pie Chart

        if st.checkbox("Pie Chart"):
            all_col_names = df.columns.to_list()
            if st.button("Generate Pie Chart"):
                st.success("Generating Pie Chart")
                st.write(df.iloc[:,-1].value_counts().plot.pie(autopct = "%1.1f%%"))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

        st.caption('Customizable Plots :sunglasses:')
    
        # Creating Customizable Plots
        all_col_names = df.columns.to_list()
        type_of_plot = st.selectbox("Select the type of Plot",["area","bar","line","hist","box","kde"])
        selected_col_names = st.multiselect("Select the Columns to Plot",all_col_names)

        if st.button("Generate Plot"):
            st.success("Generating the Customized {} Plot for {}".format(type_of_plot,selected_col_names))

            # Plots by Streamlit
            if type_of_plot == 'area':
                cust_data = df[selected_col_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = df[selected_col_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line':
                cust_data = df[selected_col_names]
                st.line_chart(cust_data)

            # Custom Plot
            elif type_of_plot:
                cust_plot = df[selected_col_names].plot(kind= type_of_plot)
                st.write(cust_plot)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

    else:

        # Outputting the Selected File
        st.write("filename:", uploaded_file.name)

        # Reading the Data
        df_uploaded = pd.read_csv(uploaded_file)

        # Showing the Dataset
        if st.checkbox("Show Dataset"):
            number = st.number_input("Number of Rows to View",5,100)
            st.dataframe(df_uploaded.head(number))

        st.subheader("Data Level Exploratory Data Analysis(EDA)")

        # Showing the Description of the Dataset
        if st.checkbox("Description of the Dataset : Some Basic Statistics"):
            st.write(df_uploaded.describe().T)

        # Showing the Columns
        if st.checkbox("Column Names"):
            st.write(df_uploaded.columns)
    
        # Showing DataTypes of the Columns
        if st.checkbox("Data Types"):
            st.write(df_uploaded.dtypes)

        # Select Columns to Show
        if st.checkbox("Select Columns to Show"):
            all_cols = df_uploaded.columns.to_list()
            selected_cols = st.multiselect("Select Columns",all_cols)
            new_df = df_uploaded[selected_cols]  
            st.dataframe(new_df)

        # Vlaue Counts per Selected Column    
        if st.checkbox("Value Counts"):
            cols = df_uploaded.columns.to_list()
            col = st.selectbox("Select a Column",cols)
            st.text("The Value Counts of the Selected Columns are as follows :")
            st.write(df_uploaded[col].sort_values(ascending=False).value_counts())

        # Showing the Shape of Dataset
        if st.checkbox("Shape of the Dataset"):
            st.write(df_uploaded.shape)
            data_dim = st.radio("Show Dimensions by ",("Rows","Columns"))
            if data_dim == 'Rows':
                st.text("Number of Rows")
                st.write(df_uploaded.shape[0])
            elif data_dim == 'Columns':
                st.text("Number of Columns")
                st.write(df_uploaded.shape[1])
            else:
                st.write(df_uploaded.shape)

        # Showing the Null Values by Column
        if st.checkbox("Checking For Null Values"):
            st.write(df_uploaded.isna().sum())

        st.subheader("Data Visualization : Visual Level EDA")

        # Count Plots

        if st.checkbox("Plot of Value Counts"):
            st.text("Value Counts by Target")
            all_col_names = df_uploaded.columns.to_list()
            primary_col = st.selectbox("Primary Column to GroupBy", all_col_names)
            selected_col_names = st.multiselect("Select Columns", all_col_names)
            if st.button("Plot"):
                st.text("Generate Plot")
                if selected_col_names:
                    vc_plot = df_uploaded.groupby(primary_col)[selected_col_names].count()
                else:
                    vc_plot = df_uploaded.iloc[:,-1].value_counts()
                st.write(vc_plot.plot(kind="bar"))
                st.pyplot()

        # Seaborn Plots

        if st.checkbox("Correlation Plot[using Seaborn]"):
            st.write(sns.heatmap(df_uploaded.corr(), annot=True))
            st.pyplot()

        # Pie Chart

        if st.checkbox("Pie Chart"):
            all_col_names = df_uploaded.columns.to_list()
            if st.button("Generate Pie Chart"):
                st.success("Generating Pie Chart")
                st.write(df_uploaded.iloc[:,-1].value_counts().plot.pie(autopct = "%1.1f%%"))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()


        st.caption('Customizable Plots :sunglasses:')
    
        # Creating Customizable Plots
        all_col_names = df_uploaded.columns.to_list()
        type_of_plot = st.selectbox("Select the type of Plot",["area","bar","line","hist","box","kde"])
        selected_col_names = st.multiselect("Select the Columns to Plot",all_col_names)

        if st.button("Generate Plot"):
            st.success("Generating the Customized {} Plot for {}".format(type_of_plot,selected_col_names))

            # Plots by Streamlit
            if type_of_plot == 'area':
                cust_data = df_uploaded[selected_col_names]
                st.area_chart(cust_data)

            elif type_of_plot == 'bar':
                cust_data = df_uploaded[selected_col_names]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line':
                cust_data = df_uploaded[selected_col_names]
                st.line_chart(cust_data)

            # Custom Plot
            elif type_of_plot:
                cust_plot = df_uploaded[selected_col_names].plot(kind= type_of_plot)
                st.write(cust_plot)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

    # Information About the Application
    st.sidebar.header("About App")
    st.sidebar.info("A Simple Exploratory Data Analysis(EDA) App for Exploring Common Machine Learning Datasets")

    st.sidebar.header("Get Datasets")
    st.sidebar.markdown("[Common ML Dataset Repo](https://github.com/RoyMustang-Dev/ML-Dataset-Explorer-App/tree/main/datasets)")

    st.sidebar.header("About")
    st.sidebar.info("Aditya Mishra")
    st.sidebar.text("Built with Streamlit")
    st.sidebar.text("Maintained by Aditya Mishra")

    if st.button("Thanks"):
	    st.balloons()

if __name__ == '__main__':
    main()