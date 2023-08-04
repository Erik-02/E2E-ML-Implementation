import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import logger, variables


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

prestage_setup()

# Import data from database
# Ensures that function only runs once and not for every user interaction
st.cache_data   
def collect_data(engine):
    print('fetching data')

    # Connect to the database
    # Connect to the database
    engine = create_engine(engine)
        

    # Fetch all data
    Logger.info('Fetching all data from database.')
    retrieve = "SELECT * FROM crude_oil_imports"
    past_data = pd.read_sql(retrieve,engine)  # Data is now in a dataframe
    Logger.info('Data retrieved and ready for use.')

    # Fetch all data
    Logger.info('Fetching all data from database.')
    retrieve = "SELECT * FROM predictions"
    predictions_data = pd.read_sql(retrieve,engine)  # Data is now in a dataframe
    Logger.info('Data retrieved and ready for use.')

    # Return the 2 dataframes containing our retrieved data from database
    return past_data, predictions_data


past_data, predictions_data = collect_data("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=DB_HOSTNAME, db=DB_DATABASE, user=DB_USERNAME, pw=DB_PASSWORD))

# Transform list of data retrieved from database to dataframe
Logger.info('Transforming data to dataframe.')
# Drop duplicates
past_data.drop_duplicates(subset=['period'], inplace=True)

# Transform list of data retrieved from database to dataframe
Logger.info('Transforming data to dataframe.')
# Drop duplicates
predictions_data.drop_duplicates(subset=['period'], inplace=True)
predictions_data.index = [i + past_data.shape[0] for i in predictions_data.index]


def plot(previous_df, predictions_df, column):

    # Get standard deviation of the predictions
    std = predictions_df[column].std()

    df1 = predictions_df.assign(upper_std=lambda x: x[column] + std)
    df1['lower_std'] = predictions_df[column].values - std

    global fig
    fig = plt.figure(figsize=(20,10))

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
    


Logger.info('Streamlit app process about to begin.')

# Selectbox to select which type of oil to display
option = st.selectbox('SELECT COLUMN DATA TO DISPLAY',
    ('heavy_sour', 'heavy_sweet', 'light_sour', 'light_sweet', 'medium', 'total'))

# call function which creates the plot to see.
plot(past_data, predictions_data, option)

# Display figure on webapp
st.pyplot(fig=fig)

# Display predictions dataframe to show exact values of predictions
st.dataframe(predictions_data[['period',option]])
Logger.info('Streamlit app process completed.')
