# Import necessary libraries
# Basic usage libraries
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
import logger, variables
# Database library
import pymysql
# Model libraries
from prophet import Prophet
from ThymeBoost import ThymeBoost as tb
# Boxcox transformations
from scipy.stats import boxcox
from scipy.special import inv_boxcox
# Train test split
from sktime.forecasting.model_selection import temporal_train_test_split
# Metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
# Weights and Biases
import wandb
# Streamlit webapp
import streamlit as st


# STEP 0: Environment setup
# Setting up environment variables and Logger
def prestage_setup():
    global Logger, EIA_KEY, WANDB_KEY, DB_HOSTNAME, DB_DATABASE, DB_USERNAME, DB_PASSWORD

    # Set up Logger
    Logger = logger.get_logger(__name__)
    Logger.info('Logger setup completed.')

    # Load API keys from variables
    Logger.info('Loading environment variables.')
    variables.load_variables()
    Logger.info('Environment variables loaded successfully.')

    # Get API keys
    Logger.info('Loading specific API keys.')
    EIA_KEY = variables.EIA_KEY
    WANDB_KEY = variables.WANDB_KEY
    DB_HOSTNAME = variables.DB_HOSTNAME
    DB_DATABASE = variables.DB_DATABASE
    DB_USERNAME = variables.DB_USERNAME
    DB_PASSWORD = variables.DB_PASSWORD
    Logger.info('API keys loaded.')


# STEP 1: Extract
# Create function to extract the data from the API
def extract():
    Logger.info('Start of extract phase.')

    # Date processing
    Logger.info('Creating extraction time variables.')
    current_date = date.today() # Get current date, Date when function is called. 1st of each month.
    past_date  = current_date - relativedelta(months=3) # Get date of 3 months before, since we have a 3 month delay in our data API
    past_date = past_date.strftime('%Y-%m') # Convert date to string, necessary for the request
    Logger.info('Extraction time variables created.')

    # Url creation
    Logger.info('Creating URL to retrieve data.')
    base_url = f'https://api.eia.gov/v2/crude-oil-imports/data/?frequency=monthly&data[0]=quantity&start={past_date}&end={past_date}&sort[0][column]=period&sort[0][direction]=desc&offset=0&api_key='
    URL = base_url + (EIA_KEY)
    Logger.info('URL ready to be used.')

    # Get Data
    Logger.info('Data retrieval about to start.')
    try:
        Logger.info('Retrieving API data.')
        response = requests.get(URL)
        Logger.info('Successfully retrieved data from API.')
    except:
        Logger.exception('Could not retrieve API data')

    Logger.info('Extraction phase about to end.')
    return response


# STEP 2: Transform
# Transform data into correct and useable format.
def transform(response):
    Logger.info('Start of transformation process.')

    # Extract the individual data records. each record is stored as a dictionary, 
    # where all of the dictionaries/entries forms the list of entries
    Logger.info('Transforming JSSON data to list.')
    data_list = ((response.json())['response']['data'])
    Logger.info('JSON data now in list format.')

    # Now that we have the list of dictionaries containing our data entries, we can convert this data into a dataframe
    Logger.info('Transforming list to DataFrame.')
    df = pd.DataFrame(data_list)
    Logger.info('Data Transformed into DataFrame.')

    Logger.info('Performing additional transformations.')
    # Extract the quantity of barrels based on each gradeName for each month.
    gpb_obj = df.groupby(['period','gradeName'])['quantity'].sum()
    # Transform groupby object into the correct dataframe
    finaldf = gpb_obj.to_frame().reset_index()
    # Pivot table into a more compact DataFrame
    finaldf = finaldf.pivot(index='period', columns='gradeName', values='quantity')
    # Remove unnecessary index and column names that resulted from the pivot table
    finaldf.columns = [''.join(str(s).strip() for s in col if s) for col in finaldf.columns]
    finaldf.reset_index(inplace=True)

    # Add a 'Total' column
    finaldf['Total'] = finaldf[['HeavySour', 'HeavySweet', 'LightSour', 'LightSweet', 'Medium']].sum(axis=1)
    # Add 'extraction_date' column
    finaldf['extraction_date'] = date.today().strftime('%Y-%m')

    Logger.info('End of Transformation process.')

    return finaldf


# STEP 3: Upload and Fetch
# Upload latest data to online database
def upload_fetch_data(df):

    Logger.info('Creating variables to be uploaded to online database.')
    # Create variables that will be uploaded
    period = df.iloc[0]['period']
    heavy_sour = int(df['HeavySour'].values)
    heavy_sweet = int(df['HeavySweet'].values)
    light_sour = int(df['LightSour'].values)
    light_sweet =  int(df['LightSweet'].values)
    medium = int(df['Medium'].values)
    total = int(df['Total'].values)
    extraction_date = df.iloc[0]['extraction_date']

    Logger.info('Trying to connect to online database.')
    try:
        # Connect to the database
        connection = pymysql.connect(host=DB_HOSTNAME,
                                user=DB_USERNAME,
                                password=DB_PASSWORD,
                                db=DB_DATABASE, 
                                autocommit=True)
        
        # Create cursor
        cursor = connection.cursor()
            
        # Insert latest record
        Logger.info('Inserting latest data to the online database.')
        # Create a new record
        sql = "INSERT INTO crude_oil_imports (period, heavy_sour, heavy_sweet, light_sour, light_sweet, medium, total, extraction_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        # Execute the query
        cursor.execute(sql, (period, heavy_sour, heavy_sweet, light_sour, light_sweet, medium, total, extraction_date))
        Logger.info('Data uploaded to database.')
            
        # Fetch all data
        Logger.info('Fetching all data from database.')
        cursor.execute("SELECT * FROM crude_oil_imports")
        all_data = cursor.fetchall()  # Data is now in the form of a list
        Logger.info('Data retrieved and ready for use.')

    except:
        Logger.exception('Could not log in to online database.')

    # Transform list of data retrieved from database to dataframe
    Logger.info('Transforming data to dataframe.')
    all_data = pd.DataFrame(all_data, columns=['period','heavy_sour', 'heavy_sweet', 'light_sour', 'light_sweet','medium', 'total', 'extraction_date'])
    # Drop duplicates
    all_data.drop_duplicates( subset=['period'], inplace=True)


    Logger.info('Data uploading and retrieval completed.')

    return all_data


# STEP 4: Outlier removal
# Remove and replace outliers within the data using FBprophet
def replace_outliers(df):

    Logger.info('Creating no outlier dataframe.')
    # Create an empty dataframe which will take in the new values from after after the outlier detection process.
    no_outlier_df = pd.DataFrame(df['period'].copy(), columns=['period'])
    #no_outlier_df['period'] = pd.to_datetime(no_outlier_df['period'])
    Logger.info('No outlier dataframe created successfully.')


    """
    Create a for loop which will take in each column, one at a time, 
    and give all data to the FBprophet model to achieve an accurate fit on this data, the model will then predict
    which datapoints are outside of the 95% confidence interval. We will then take these datapoints that are outside of the 
    confidence interval and replace them with the model's predicted values.
    """
    Logger.info('Start of "For loop" which will replace outliers with the predicted values.')
    for column in df.columns[1:-1]: # essentially, we only want the volumns of crude oil, not the time related columns
        
        Logger.info('Transforming column data into correct format for FBprophet model.')
        # Create special dataframe with the period and specified column that will be sued for FBprophet outlier detection model
        outlier_data = pd.DataFrame(df['period'].copy())                            # get period
        outlier_data['period'] = pd.to_datetime(outlier_data['period'])                             # make period a Datetime object
        outlier_data[column] = df[column].copy()                                    # get column data
        outlier_data.rename(columns={'period':'ds', column:'y'}, inplace=True)      # rename columns for FBprophet
        
        # Boxcox data in order to get normalize the data and achieve more accurate results
        Logger.info('Transforming data with boxcox.')
        outlier_data['y'], lmbda = boxcox(outlier_data['y'])
        
        Logger.info('Start of model building.')
        # Create FBporphet model with 95% confidence level, a.k.a. ,2 Standard deviations
        fbp_model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=False)
        
        Logger.info('Fitting model to data.')
        # Fit model to data
        fbp_model.fit(outlier_data)

        # forecast on trained data, this will reveal which data points are outside of the
        Logger.info('Creating predictions with model on given data.') 
        heavy_sour_forecast = fbp_model.predict(outlier_data)
        
        # Merge actual and predicted dataframes
        Logger.info('Merging predicted and actual dataframes.')
        performance = pd.merge(outlier_data, heavy_sour_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        # Create anomaly indicator, if actual value is higher or lower than the predicted bounds, it is an anomaly
        Logger.info('Creating anomaly indicators on the data.')
        performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)

        # Replace the actual values with the predicted values, only where the data is indicated as an anomaly
        Logger.info('Replacing anomalies with predicted values')
        performance['y'].mask(performance['anomaly']==1, performance['yhat'], inplace=True)
        
        # Reverse the boxcox transformation to use original data again
        Logger.info('Inverting Boxcox transform')
        performance['y'] = inv_boxcox(performance['y'], lmbda)
        
        # The actual data does not have any outliers anymore, and we can use this data to proceed
        Logger.info('Assigning no-outlier data to no_outlier_df.')
        no_outlier_df[column] = performance['y'].copy()

    Logger.info('End of outlier removal process.')

    return no_outlier_df


# STEP 5: ThymeBoost
""" Create a function to loop through all of the oil columns and create a training and testing set.
Use this training and testing set to get accuracy scores, such as the MAE and MAPE. 
Then, upload these scores as a model run in Weights and Biases.
Thereafter, use entire columns data to train a model and make 12 predictions, these predictions are essentially the next 12 months predictions.
Save and upload both this column data on which model is trained as well as the predictions dataset to Weights and Biases.
As long as we have the training data saved, we do not need to save the model, 
since a ThymeBoost model chooses the best fit and will remain the same as long as the data is the same.
""" 
def model_training(df):

    # Initialize future dataframe with future period values
    Logger.info('Initializing prediction dataframe.')
    predicted_df = pd.DataFrame({'period': pd.date_range(start=df.period.iloc[-1], periods= 12, freq='MS')+ pd.DateOffset(months=1)})

    # Login to Weights and Biases
    Logger.info('Logging in to Weights and Biases.')
    wandb.login(relogin=True, key=WANDB_KEY)
    Logger.info('Successfully logged in to Weights and Biases.')
    

    Logger.info('Model training loop about to start.')
    for column in df.columns[1:]:

               
        # Time of process start
        dt1 = datetime.datetime.now()

        # Create train test split
        Logger.info(f'creating train test split with 15% testing data from column {column}.')
        train, test = temporal_train_test_split(df[['period', column]], test_size=0.15)
        
        # Initiate thymeboost model
        Logger.info('Initiating ThymeBoost model.')
        boosted_model = tb.ThymeBoost(approximate_splits=False, verbose=0, cost_penalty=.001)

        # Fit thymeboost model to training data
        Logger.info('Fitting ThymeBoost model to training data.')
        output = boosted_model.autofit(train[column], 
                                        seasonal_period=[3,4,6,12])

        # Make predictions with model
        Logger.info('Making predictions on testing data.')
        predicted_output = boosted_model.predict(output, len(test))
        predicted_output.index = test.index.copy()

        # Get time when training stopped
        dt2 = datetime.datetime.now()
        
        # Get model performance accuracy
        Logger.info(f'Obtaining {column} Accuracy scores, MAE & MAPE and Training time.')
        MAE = mean_absolute_error(test[column] , predicted_output["predictions"])
        MAPE = mean_absolute_percentage_error(test[column] , predicted_output["predictions"])
        training_time = str(dt2-dt1)
        
        
        # Write accuracies to Weights and Biases
        Logger.info(f'Logging accuracy scores of {column} to Weights and Biases.')
        # 🐝 1️⃣ Start a new run to track this script
        run = wandb.init(
        # Set the project where this run will be logged
        project="crude_oil_imports", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{column}", )

        wandb.log({'MAE': MAE,
                'MAPE': MAPE,
                'Train_Time': training_time})
        Logger.info(f'{column} Accuracy scores logged successfully.')
        

        # Fit all data to make predictions
        # Initiate thymeboost model
        Logger.info('Creating ThymeBoost model that will be used for Forecasting.')
        boosted_model = tb.ThymeBoost(approximate_splits=False, verbose=0, cost_penalty=.001)

        # Fit thymeboost model to training data
        Logger.info(f'Fitting model on entire {column} column.')
        output = boosted_model.autofit(df[column], seasonal_period=[3,4,6,12])
        
        # Make 12 predictions with model. Predicting next 12 month's worth of values
        Logger.info(f'Making 12 predictions for {column} column.')
        predicted_output = boosted_model.predict(output, 12)
        predictions = predicted_output['predictions'].copy()

        # Create artifacts to upload.
        # The artifacts are: 
        #       1. Training data
        """ It is not needed to save the Model or predictions, since Thymeboost autofit will select the best possible model every time. 
        Therefore as long as the input data remains the same, the model will also remane the same.
        This in turn will then predict the exact same result every time, since we have the same input data and model."""
                
        # 1. Training data artifact
        # Initialize artifact
        Logger.info(f'Creating {column} training data artifact.')
        artifact = wandb.Artifact(name = f'{column}_training_dataset', type = "data", description = f'History of {column} data')
        # Create training dataset file
        Logger.info(f'Creating {column} training dataset.')
        df[['period',column]].to_csv(f'{column}_training_dataset.gzip', index=False)
        # Add training dataset file to artifact
        Logger.info(f'Loading {column} training dataset.')
        artifact.add_file(f'{column}_training_dataset.gzip')
        # Log artifact
        Logger.info(f'Adding {column} training data artifact to Weights and Biases.')
        run.log_artifact(artifact)


        # Mark the run as finished
        Logger.info(f'Finishing wandb run of {column}.')
        wandb.finish()

        # Adding predictions to predictions dataframe
        Logger.info(f'Adding {column} predictions to prediction dataframe.')
        predicted_df[column] = predictions.values

    Logger.info('End of model training/prediction process.')
    return predicted_df


# STEP 6: Plotting
# Create the plotting function for plot to be displayed on streamlit webapp.
def plot(previous_df, predictions_df, column):

    # Get standard deviation of the predictions
    std = predictions_df[column].std()

    # Create upper and lower bounds from standard deviation of the data
    df1 = predictions_df.assign(upper_std=lambda x: x[column] + std)
    df1['lower_std'] = predictions_df[column].values - std

    # Initialize global figure that will be displayed on streamlit app.
    global fig
    fig = plt.figure(figsize=(20,10))
    # Draw plot
    plt.plot(previous_df[column].index[130:], previous_df[column].values[130:], color='green', label='Past')
    plt.plot(predictions_df[column].index, predictions_df[column].values, color='Blue', label='Predictions')
    plt.fill_between(x=df1.index, y1=df1.upper_std, y2=df1.lower_std, 
    where= df1.upper_std > df1.lower_std, facecolor='purple', alpha=0.1, interpolate=True,
                        label='Standard deviation')

    # Set plot attributes
    plt.xlabel('Months')
    plt.ylabel('Average daily imports (measured in million)')
    plt.title(f'Average number of {column} crude oil imports')
    plt.legend()
    plt.grid()


# Start of workflow:
# STEP 0: CONFIG
prestage_setup()
Logger.info('Prestage setup completed.')

# STEP 1: Extract
Logger.info('Extraction process about to begin.')
response = extract()
Logger.info('Extraction process completed.')

# STEP 2: Transform
Logger.info('Transformation process about to begin.')
finalDF = transform(response)
Logger.info('Transformation process completed.')

# STEP 3: Upload and Fetch
Logger.info('Uploading of new data and retrieval of all data about to begin.')
all_data = upload_fetch_data(finalDF)
Logger.info('Uploading of new data and retrieval of all data completed.')

# STEP 4: Outlier removal
Logger.info('Outlier removal process about to begin.')
no_outliers = replace_outliers(all_data)
Logger.info('Outlier removal process completed.')

# STEP 5: Model training
Logger.info('Model training process about to begin.')
predictions = model_training(no_outliers)
Logger.info('Model training process completed.')

# STEP 6: Streamlit app
Logger.info('Streamlit app process about to begin.')

# Selectbox to select which type of oil to display
option = st.selectbox('SELECT COLUMN DATA TO DISPLAY',
    ('heavy_sour', 'heavy_sweet', 'light_sour', 'light_sweet', 'medium', 'total'))

# call function which creates the plot to see.
plot(no_outliers, predictions, option)

# Display figure on webapp
st.pyplot(fig=fig)

# Display predictions dataframe to show exact values of predictions
st.dataframe(predictions[['period',option]])
Logger.info('Streamlit app process completed.')