from django.shortcuts import render,HttpResponse

##importing the requires libraries for automate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

from sklearn.svm import SVR,SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report,mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from ydata_profiling import ProfileReport
import io
# import langchain
from bs4 import BeautifulSoup
## PandasAI
import pandasai
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm import OpenAI
from Secret_key.constants import openai_key
import matplotlib.pyplot as plt
import re
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.exceptions import ValidationError
from django.views.generic import View
from django.utils.decorators import method_decorator
from django.views.decorators.clickjacking import xframe_options_exempt
import io
from wsgiref.util import FileWrapper


# Instantiate a LLM
llm = OpenAI(api_token=openai_key)

# Define the directory where your model files are stored
# MODEL_DIRECTORY = '/home/AutoML/'

# preprocessing function
def extract_numerical_value(text):
    # Use regular expressions to extract the numerical value
    match = re.search(r'(\d+\.\d+)', str(text))
    if match:
        return float(match.group(1))  # Convert the extracted text to a float
    else:
        return None  # Return None if no match is found


def preprocessing(df, target_column=None):
    """
    Preprocesses the given DataFrame by handling missing values and converting data types.

    Args:
        df (pd.DataFrame): The input DataFrame to be preprocessed.
        target_column (str): The name of the target column (if applicable).

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if df.isnull().values.any():
        print("Null values exist in the DataFrame!")

        # Identify columns with object dtype and numeric dtype
        object_columns = df.select_dtypes(include=['object']).columns
        numeric_columns = df.select_dtypes(include=['int', 'float']).columns

        # Fill null values in object columns with the mode
        for col in object_columns:
            mode_val = df[col].mode().iloc[0]
            df[col].fillna(mode_val, inplace=True)

        # Impute missing values in numeric columns with the mean
        for col in numeric_columns:
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))

        # Identify object columns that contain numeric values with commas
        columns_with_commas = [col for col in object_columns if df[col].str.contains(',').any()]

        # Remove commas from columns with numeric values
        for col in columns_with_commas:
            df[col] = df[col].str.replace(',', '')

        if target_column:
            if target_column in object_columns:
                # Convert the target column to numeric format
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

        # Identify object columns with all numerical values and try to convert them
        for col in object_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                pass  # Ignore columns that can't be converted to numeric

    return df

# Modify the classify and regress functions to return the trained model object and its name
# def classify(X_train_final, y_train, X_test_final, y_test):
#     # Define the classification model you want to use
#     model = RandomForestClassifier()

#     # Perform cross-validation to evaluate the model
#     scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1_macro')
#     average_f1 = scores.mean()

#     # Train the model on the entire training dataset
#     model.fit(X_train_final, y_train)

#     # Generate a classification report for the model
#     y_pred = model.predict(X_test_final)
#     report = classification_report(y_test, y_pred, output_dict=True)

#     return model, "RandomForestClassifier", average_f1, report

def classify(X_train_final, y_train, X_test_final, y_test):
    algorithms = {
        "RandomForestClassifier": RandomForestClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(),
        "XGBClassifier": XGBClassifier(),
        "SVC":SVC(),
        "KNN":KNeighborsClassifier(),
        "BerunoulliNB":BernoulliNB(),
        "GassuainNB":GaussianNB(),
    }

    accuracy_scores = []  # Create a list to store accuracy scores

    best_model = None
    best_model_name = ""
    best_accuracy = 0
    best_report = None

    for name, model in algorithms.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append((name, accuracy))  # Append (model name, accuracy) tuple
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
            best_report = classification_report(
                y_test, y_pred, output_dict=True)
    
    print("list of score:- ")    
    for i in accuracy_scores:
        print(i)    
    return best_model, best_model_name, best_accuracy, best_report


def regress(X_train_final, y_train, X_test_final, y_test):
    algorithms = {
        "RandomForestRegressor": RandomForestRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "LinearRegression": LinearRegression(),
        "XGBRegressor": XGBRegressor()
    }

    best_model = None
    best_model_name = ""
    best_mse = float("inf")
    best_mae = float("inf")
    best_rmse = float("inf")
    best_predictions = None

    for name, model in algorithms.items():
        # Identify columns with missing values in training data
        missing_cols = X_train_final.columns[X_train_final.isnull().any()]

        # Create a list of transformers for the pipeline
        transformers = []

        # Add the model to the pipeline
        transformers.append(('model', model))

        # Create the pipeline
        pipeline = Pipeline(transformers)

        pipeline.fit(X_train_final, y_train)
        y_pred = pipeline.predict(X_test_final)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_rmse = rmse
            best_model = pipeline
            best_model_name = name
            best_predictions = y_pred

    return best_model, best_model_name, best_mse, best_mae, best_rmse, best_predictions

def prediction(df, target_column, problem_type, test_size=0.3, random_state=42):
    # Split the DataFrame into features (X) and the target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Perform one-hot encoding for categorical features
    X = pd.get_dummies(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    if problem_type == 'Classification':
        if y.dtype == 'object':
            # If the target column is categorical, perform label encoding
            lb = LabelEncoder()
            y_train = lb.fit_transform(y_train)
            y_test = lb.transform(y_test)
        
        # Call the classify function to train and evaluate the RandomForestClassifier
        best_model, best_model_name, best_accuracy, report = classify(
            X_train, y_train, X_test, y_test)

        return best_model, best_model_name, best_accuracy, report

    elif problem_type == 'Regression':
        # Call the regress function to train and evaluate the RandomForestRegressor
        best_model, best_model_name, best_mse, best_mae, best_rmse, predictions = regress(
            X_train, y_train, X_test, y_test)

        return best_model, best_model_name, best_mse, best_mae, best_rmse, predictions

    else:
        # Handle the case where an invalid problem type is provided
        raise ValueError("Invalid problem type. Use 'Classification' or 'Regression'.")


global_df = None
# Create your views here

def home(request):
    problem_type = None
    target_column = None

    try:
        if request.method == 'POST':
            if 'csvFile' not in request.FILES:
                return HttpResponse("No file part")

            uploaded_file = request.FILES['csvFile']

            # Check the file size
            max_file_size = 2 * 1024 * 1024  # 2MB
            if uploaded_file.size > max_file_size:
                return HttpResponse("File size exceeds the limit of 2MB.")

            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension.lower() not in ['csv', 'xlsx', 'xls']:
                return HttpResponse("Invalid file format. Please upload a CSV or Excel file.")

            if file_extension.lower() == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                # For Excel files (xlsx or xls), you can use pd.read_excel
                df = pd.read_excel(uploaded_file)

            global global_df
            global_df = df

            # Preprocess the DataFrame
            preprocessed_df = preprocessing(df)
            top_5 = preprocessed_df
            # print(top_5)

            if preprocessed_df is not None:
                if 'problemType' in request.POST and 'targetcol' in request.POST and 'name' in request.POST:
                    problem_type = request.POST['problemType']
                    target_column = request.POST['targetcol']

                    # Check if the target column exists in the DataFrame
                    if target_column not in preprocessed_df.columns:
                        raise ValidationError(f"Target column '{target_column}' does not exist in the dataset. Please check your dataset.")

                    # Check if the selected problem type matches the data type of the target column
                    if problem_type == 'Classification' and preprocessed_df[target_column].dtype not in ['object', 'int64', 'float64']:
                        raise ValidationError(f"Selected problem type is Classification, but the target column '{target_column}' contains non-categorical data. Please select the correct problem type.")
                    elif problem_type == 'Regression' and preprocessed_df[target_column].dtype not in ['int64', 'float64']:
                        raise ValidationError(f"Selected problem type is Regression, but the target column '{target_column}' contains non-numeric data. Please select the correct problem type.")

                    project_title = request.POST['name']

                    if problem_type == 'Classification':
                        result = prediction(
                            preprocessed_df, target_column, problem_type)
                        if isinstance(result, tuple) and len(result) == 4:
                            best_model, best_model_name, best_accuracy, report = result
                            # print(report)
                            
                            # Save the best classification model
                            # model_directory = os.path.join(settings.MEDIA_ROOT, 'models')
                            model_filename = f"{project_title}_best_classification_model.joblib"
                            # model_filepath = os.path.join(model_directory, model_filename)
                            joblib.dump(best_model, model_filename)

                            # Provide a download link for the model
                            model_link = f"/download/{model_filename}"

                            # Pass the best model name and other information to the template
                            return render(request, 'home.html', {
                                'table': top_5,
                                'problem_type': problem_type,
                                'target_column': target_column,
                                'project_title': project_title,
                                'prediction_report': report,
                                'accuracy': best_accuracy * 100,
                                'model_link': model_link,
                                "report":report,
                                'best_model_name': best_model_name
                            })

                    elif problem_type == 'Regression':
                        print(preprocessed_df)
                        best_model, best_model_name, best_mse, best_mae, best_rmse, predictions = prediction(
                            preprocessed_df, target_column, problem_type)
                        
                        print(predictions)
                        # Save the best regression model
                        model_filename = f"{project_title}_best_regression_model.joblib"
                        joblib.dump(best_model, model_filename)

                        # Provide a download link for the model
                        model_link = f"/download/{model_filename}"

                        # Pass the best model name and other information to the template
                        return render(request, 'home.html', {
                            'table': top_5,
                            'problem_type': problem_type,
                            'target_column': target_column,
                            'project_title': project_title,
                            'mse': best_mse,
                            'mae': best_mae,
                            'rmse': best_rmse,
                            'predictions': predictions,
                            'model_link': model_link,
                            'best_model_name': best_model_name
                        })
                    else:
                        return HttpResponse("Other ML Algorithms are not implemented yet. Only Regression and Classification problems can be handled by this application.")
        return render(request, 'home.html', {
            'table': None,
            'error_message': None,
            'prediction_report': None,
            'problem_type': problem_type,
            'target_column': target_column
        })
    except Exception as e:
        error_message = str(e)
        return render(request, 'home.html', {
            'table': None,
            'error_message': error_message,
            'prediction_report': None,
            'problem_type': problem_type,
            'target_column': target_column
        })

## Chat with datasets
def chat_datasets(request):
    question = None
    error_message = None
    # table = None
    context = {}  # Initialize context

    if request.method == "POST":
        question = request.POST.get('question')

    try:
        global global_df  # Access the global DataFrame
        if global_df is None:
            return HttpResponse("No dataset available. Please upload a CSV file first.")

        df = SmartDataframe(global_df, config={"llm": llm})
        if question:
            result = df.chat(question)
        print(global_df)
        return render(request, "chat.html", {
                "table":global_df,
                "result": result,
                "question": question,
            })
        
    except Exception as e:
        error_message = "Something went wrong 😒😔!"  # Handle exceptions here
        print(global_df)
    return render(request, "chat.html", {
            "table": global_df,
            "message": error_message,
        })

report_html = None  # Initialize report_html at the module level
def generate_and_display_profile(request):
    global report_html  # Use the global report_html variable
    profile = None
    report_generated = False  # Add a variable to track report generation

    if request.method == 'POST':
        if 'csvFile' not in request.FILES:  # Note the change here to request.FILES
            return HttpResponse("No file part")

        file = request.FILES['csvFile']  # Note the change here to request.FILES

        if file.name == '':
            return HttpResponse("No selected file")

        if not file.name.endswith('.csv'):
            return HttpResponse("Invalid file format. Please upload a CSV file.")

        # Check the file size
        max_file_size = 2 * 1024 * 1024  # 2MB
        if file.size > max_file_size:
            return HttpResponse("File size exceeds the limit of 2MB.")

        df = pd.read_csv(file)
        # Create a Pandas Profiling Report
        profile = ProfileReport(df, title="DataSet's Report")

        # Convert the report to HTML as a string
        report_html = profile.to_html()
        report_generated = True  # Set report_generated to True when the report is generated

    if report_html:
        # Option 1: Render the report HTML
        return render(request, "generated_report.html", {"report_html": report_html, "report_generated": report_generated})
    
    elif global_df is not None and not global_df.empty:
        profile = ProfileReport(global_df, title="DataSet's Report")
        report_html = profile.to_html()
        report_generated = True  # Set report_generated to True when the report is generated
        # Option 1: Render the report HTML
        return render(request, "generated_report.html", {"report_html": report_html, "report_generated": report_generated})
    # else:
    #     return HttpResponse("Kindly upload the file to generate the report ")
    # Option 2: Provide a download link for the generated report
    return render(request, 'datasets_report.html', {"report_generated": report_generated})


@method_decorator(xframe_options_exempt, name='dispatch')
class DownloadReportView(View):
    def get(self, request):
        global report_html  # Use the global report_html variable
        if report_html is not None:
            # Create an in-memory file object to store the HTML content
            report_file = io.BytesIO()
            report_file.write(report_html.encode('utf-8'))
            report_file.seek(0)

            # Set response headers to force download
            response = HttpResponse(report_file, content_type='text/html')
            response['Content-Disposition'] = 'attachment; filename="report.html"'

            return response

        return HttpResponse("Report not found")




def download_model(request, model_filename):
    try:
        # Create the full file path
        file_path = os.path.abspath(model_filename)
        
        # Check if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'rb') as model_file:
                response = HttpResponse(FileWrapper(model_file), content_type='application/force-download')
                response['Content-Disposition'] = f'attachment; filename="{model_filename}"'
                return response
        else:
            # Handle the case where the file doesn't exist
            return HttpResponse("File not found", status=404)
    except Exception as e:
        # Handle any other exceptions that may occur during the process
        print("Error:", str(e))
        return HttpResponse("An error occurred while trying to download the file: " + str(e), status=500)