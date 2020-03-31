import numpy as np
import pandas as pd
import warnings
import re
import os
import shutil
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings('ignore')


TODAY = datetime.now().date()
START_DATE = '03-01-2020'
DAYS_AHEAD_TO_FORECAST = 7
PLOTS_DIRECTORY = 'plots'
DATA_DIRECTORY = 'data'


COUNTRIES_TO_ANALYZE = ['USA', 'CAN']
COLUMNS_TO_PLOT_TS = ['daily_percent_change_cases', 'daily_percent_change_deaths', 'cumulative_cases',
                      'cumulative_deaths', 'daily_cases', 'daily_deaths']
COLUMNS_TO_MODEL_EXPONENTIAL = ['cumulative_cases']
COLUMNS_TO_MODEL_POLYNOMIAL = ['cumulative_cases']


def make_directories_if_needed():
    """
    Creates PLOTS_DIRECTORY and DATA_DIRECTORY if they do not exist.
    """
    if not os.path.exists(os.path.join(PLOTS_DIRECTORY, str(TODAY))):
        os.makedirs(os.path.join(PLOTS_DIRECTORY, str(TODAY)))
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)


def _clear_directories_helper(directory):
    """
    Helper function that removes all files in the provided directory.
    :param directory: directory from which to remove files
    """
    paths = glob.glob(os.path.join(directory, '*'))
    for path in paths:
        if os.path.isdir(path) and os.path.exists(path):
            shutil.rmtree(path)
        elif not os.path.isdir(path) and os.path.exists(path):
            os.remove(path)


def clear_directories_if_desired(clear_data_directory=False, clear_plots_directory=False):
    """
    Clears DATA_DIRECTORY and / or PLOTS_DIRECTORY if desired.
    :param clear_data_directory: boolean of whether or not to clear contents of DATA_DIRECTORY
    :param clear_plots_directory: boolean of whether or not to clear contents of PLOTS_DIRECTORY
    """
    if clear_data_directory:
        _clear_directories_helper(DATA_DIRECTORY)
    if clear_plots_directory:
        _clear_directories_helper(PLOTS_DIRECTORY)


def get_data():
    """
    Fetches COVID-19 data from an official EU website and writes the results to a csv file, stamped with the date. The
    file is updated once a day, per the web page. If we have already pulled the file on a given date, we will have the
    most recent copy. In such cases, we will read from the local csv file rather than pull from the remote source.
    :return: pandas dataframe containing COVID-19 daily data by country
    """
    if not os.path.exists(os.path.join(DATA_DIRECTORY, 'covid19_data_' + str(TODAY) + '.csv')):
        df = pd.read_csv('https://opendata.ecdc.europa.eu/covid19/casedistribution/csv', encoding='ISO-8859-1')
        df.to_csv(os.path.join(DATA_DIRECTORY, 'covid19_data_' + str(TODAY) + '.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'covid19_data_' + str(TODAY) + '.csv'), encoding='ISO-8859-1')
    df.rename(columns={'countryterritoryCode': 'country_code', 'cases': 'daily_cases', 'deaths': 'daily_deaths'},
              inplace=True)
    return df


def convert_camel_case_to_snake_case(df):
    """
    Converts dataframe column headers from snakeCase to camel_case.
    :param df: any valid pandas dataframe
    :return: pandas dataframe
    """
    new_columns = []
    for column in list(df):
        new_column = re.sub(r'(?<!^)(?=[A-Z])', '_', column).lower()
        new_columns.append(new_column)
    df.columns = new_columns
    return df


def ensure_date_is_sorted(df):
    """
    Reconstructs provided date format into MM-DD-YYY and ensures the data is sorted in ascending order based on the
    date.
    :param df: covid19_df defined in main()
    :return: pandas dataframe
    """
    df['date_rep'] = df['date_rep'].str[3:5] + '-' + df['date_rep'].str[0:2] + '-' + df['date_rep'].str[6:]
    df['date_rep'] = pd.to_datetime(df['date_rep'])
    df.sort_values(by=['geo_id', 'date_rep'], ascending=True, inplace=True)
    return df


def adjust_date(df):
    """
    The date_rep field represents data from the end of the previous day. This function creates a field called date,
    which represents the actual date of occurrence rather than the reported date.
    :param df: covid19_df defined in main()
    :return: pandas dataframe
    """
    df['date'] = df['date_rep'] - pd.to_timedelta(1, unit='d')
    return df


def _subset_countries(df, countries_list):
    """
    Subsets dataframe to a set of countries.
    :param df: covid19_df defined in main()
    :param countries_list: python list containing country codes (original column name is countryterritoryCode) found in
    the data provided by the EU
    :return: pandas dataframe
    """
    df = df.loc[df['country_code'].isin(countries_list)]
    return df


def _calculate_stats_helper(df):
    """
    Calculates the following statistics from the raw data: cumulative_cases, cumulative_deaths, death_rate,
    daily_percent_change_cases, and daily_percent_change_deaths.
    :param df: covid19_df defined in main()
    :return: pandas dataframe
    """
    df['cumulative_cases'] = df.groupby(['country_code'])['daily_cases'].apply(lambda x: x.cumsum())
    df['cumulative_deaths'] = df.groupby(['country_code'])['daily_deaths'].apply(lambda x: x.cumsum())
    df['death_rate'] = df['cumulative_deaths'] / df['cumulative_cases']
    df['daily_percent_change_cases'] = df['cumulative_cases'].pct_change()
    df['daily_percent_change_deaths'] = df['cumulative_deaths'].pct_change()
    df.fillna({'death_rate': 0, 'daily_percent_change_cases': 0, 'daily_percent_change_deaths': 0}, inplace=True)
    return df


def calculate_stats(df, include_worldwide_analysis=True):
    """
    Appends features generated by _calculate_stats_helper() and concatenates worldwide and country-specific dataframes
    if needed.
    :param df: covid19_df defined in main()
    :param include_worldwide_analysis: boolean of whether or not to calculate worldwide stats in addition to
    country-level stats
    :return: pandas dataframe
    """
    countries_df = _subset_countries(df, countries_list=COUNTRIES_TO_ANALYZE)
    countries_df = _calculate_stats_helper(countries_df)
    if include_worldwide_analysis:
        worldwide_df = df.groupby(['date']).agg({'daily_cases': 'sum', 'daily_deaths': 'sum'})
        worldwide_df.reset_index(inplace=True)
        worldwide_df['country_code'] = 'Worldwide'
        worldwide_df = _calculate_stats_helper(worldwide_df)
        df = pd.concat([countries_df, worldwide_df])
    return df


def make_time_series_plot(df, column, start_date, countries_list):
    """
    Makes a time-series plot.
    :param df: covid19_df defined in main()
    :param column: the column to plot
    :param start_date: the date on which to start the plot
    :param countries_list: the list of countries to analyze
    """
    for country in countries_list:
        country_df = df.loc[(df['date'] >= start_date) & (df['country_code'] == country)]
        sns.lineplot(x='date', y=column, hue='country_code', data=country_df)
        column_title = (column.replace('_', ' ')).title()
        plt.xlim(start_date, str(TODAY))
        plt.title(country + ' Time Series Plot for ' + column_title)
        plt.xticks(rotation=90)
        plt.legend(loc='upper right')
        plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
        if 'percent_change' not in column:
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIRECTORY, str(TODAY), 'time_series_plot_for_' + column + '_' + country
                                 + '.png'), bbox_inches='tight')
        plt.clf()


def _exponential_function(x, a, b, c):
    """
    Defines exponential function.
    :return: exponential function
    """
    return a * np.exp(-b * x) + c


def _create_x_tick_marks(start_date, end_days_ahead, day_increment):
    """
    Generates tick mark locations and labels for the x-axis of a time-series plot. The tick marks are spaced out based
    on the day_increment. The labels are all date values, intended to replace the existing numeric index on the x-axis.
    The labels and tick marks will begin with the start date.
    :param start_date: the date on which the plot should begin
    :param end_days_ahead: how many days ahead we want to show predictions for
    :param day_increment: how many days to place in between tick marks on the x-axis
    :return: tuple with two items: 1) python list of labels for x-axis and 2) python list for tick marks to make on
    the x-axis
    """
    start_date = datetime.strptime(start_date, '%m-%d-%Y')
    iterations = int(end_days_ahead / day_increment)
    labels = [start_date]
    date_counter = start_date
    for i in range(0, iterations):
        date_counter = date_counter + timedelta(days=7)
        labels.append(date_counter)
    labels = [label.strftime('%m-%d-%Y') for label in labels]

    tick_marks = [0]
    counter = 0
    for i in range(0, len(labels)):
        while counter < (end_days_ahead - day_increment):
            counter += day_increment
            tick_marks.append(counter)

    return labels, tick_marks


def _make_fit_and_prediction_plot(x, y, xx, yy, column, start_date, total_days, model_type, country):
    """
    Plots a fitted line and future predictions.
    :param x: training x array (predictor features)
    :param y: training y array (target values)
    :param xx: prediction x array
    :param yy: predicted y array
    :param column: the column being fit and predicted
    :param start_date: the date on which the plot should begin
    :param total_days: the total number of days we are training plus predicting on
    :param model_type: the type of model we are using, for use in the plot title
    """
    plt.plot(x, y, 'ko', label=column)
    plt.plot(xx, yy, label='Fitted Function')
    column_title = (column.replace('_', ' ')).title()
    plt.title(country + ' ' + model_type + ' Model for ' + column_title)
    plt.legend(loc='upper left')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Date')
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.ylabel(column)
    labels, tick_marks = _create_x_tick_marks(start_date, total_days, 7)
    plt.xticks(tick_marks, labels, rotation='vertical')


def _create_x_y_model_arrays(df, column, start_date, country):
    """
    Creates x and y numpy arrays for modeling.
    :param df: covid19_df defined in main()
    :param column: the series to be predicted
    :param start_date: the date on which to start modeling the series of interest
    :param country: the country we are modeling
    :return: a tuple with x and y numpy arrays, with x being a series of integers representing days and y being the
    series we want to predict
    """
    df = df.loc[(df['date'] >= start_date) & (df['country_code'] == country)]
    y = df[column]
    y = y.values
    x = np.arange(0, len(y), 1)
    return x, y


def _generate_prediction_array_and_range(y, days_ahead_to_forecast):
    """
    Creates an array of integers, representing future days, for prediction and calculates the total number of days used
    for training and prediction.
    :param y: the array of values we are interested in predicting; only used to get its length
    :param days_ahead_to_forecast: the number of days ahead we want to forecast
    :return: a tuple that includes 1) a numpy array of integers representing future days and 2) an integer that
    represents the total number of days used for training and prediction
    """
    total_days = len(y) + days_ahead_to_forecast
    xx = np.arange(0, total_days, 1)
    return xx, total_days


def model_exponential_and_forecast(df, column, start_date, days_ahead_to_forecast, countries_list):
    """
    Fits an exponential function to the series of interest, makes predictions for upcoming days, and produces a plot
    of the fitted line and predictions.
    :param df: covid19_df defined in main()
    :param column: the series to be predicted
    :param start_date: the date on which to start modeling the series of interest
    :param days_ahead_to_forecast: the number of days ahead we want to forecast
    :param countries_list: the list of countries to analyze
    """
    for country in countries_list:
        x, y = _create_x_y_model_arrays(df, column, start_date, country)
        popt, pcov = curve_fit(_exponential_function, x, y, p0=(1, 1e-6, 1))
        xx, total_days = _generate_prediction_array_and_range(y, days_ahead_to_forecast)
        yy = _exponential_function(xx, *popt)
        _make_fit_and_prediction_plot(x, y, xx, yy, column, start_date, total_days, 'Exponential', country)
        plt.savefig(os.path.join(PLOTS_DIRECTORY, str(TODAY), 'exponential_forecast_for_' + column + '_' + country
                                 + '.png'), bbox_inches='tight')
        plt.clf()


def fit_polynomial_regression_and_forecast(df, column, start_date, days_ahead_to_forecast, countries_list):
    """
    Fits a polynomial function to the series of interest, makes predictions for upcoming days, and produces a plot
    of the fitted line and predictions. The function leverages the scikit-learn framework to find the appropriate
    polynomial degree for the model. To note, the model uses no cross validation and will, therefore, overfit as the
    underlying function becomes more complex. It also uses all the data for training and does not validate on a test
    set. Such practices are not best practice. However, the focus is more on seeing a line fit rather than optimizing
    for generalization.
    :param df: covid19_df defined in main()
    :param column: the series to be predicted
    :param start_date: the date on which to start modeling the series of interest
    :param days_ahead_to_forecast: the number of days ahead we want to forecast
    :param countries_list: the list of countries to analyze
    """
    for country in countries_list:
        x, y = _create_x_y_model_arrays(df, column, start_date, country)
        x = x.reshape(-1, 1)

        pipeline = Pipeline([
            ('polynomial_features', PolynomialFeatures()),
            ('model', LinearRegression()),
        ])

        params = {'polynomial_features__degree': [2, 3, 4]}
        cv = [(slice(None), slice(None))]

        search = GridSearchCV(pipeline, params, n_jobs=1, cv=cv, verbose=0)
        search.fit(x, y)
        model = search.best_estimator_
        xx, total_days = _generate_prediction_array_and_range(y, days_ahead_to_forecast)
        xx = xx.reshape(-1, 1)
        yy = model.predict(xx)
        _make_fit_and_prediction_plot(x, y, xx, yy, column, start_date, total_days, 'Polynomial', country)
        plt.savefig(os.path.join(PLOTS_DIRECTORY, str(TODAY), 'polynomial_forecast_for_' + column + '_' + country
                                 + '.png'), bbox_inches='tight')
        plt.clf()


def main(include_worldwide_analysis=True, clear_data_directory=False, clear_plots_directory=True):
    """
    Pulls open source COVID-19 data from an EU website, sets up local directories, cleans data, and fits multiple
    simple models with results visualized.
    :param include_worldwide_analysis: boolean of whether or not to include an analysis of worldwide data;
    default is True
    :param clear_data_directory: boolean of whether or not to remove files from the DATA_DIRECTORY; default is False
    :param clear_plots_directory: boolean of whether or not to remove files from the PLOTS_DIRECTORY; default is True
    """
    clear_directories_if_desired(clear_data_directory=clear_data_directory, clear_plots_directory=clear_plots_directory)
    make_directories_if_needed()
    covid19_df = get_data()
    covid19_df = convert_camel_case_to_snake_case(covid19_df)
    covid19_df = ensure_date_is_sorted(covid19_df)
    covid19_df = adjust_date(covid19_df)
    covid19_df = calculate_stats(covid19_df, include_worldwide_analysis=include_worldwide_analysis)
    countries_list = list(set(covid19_df['country_code'].tolist()))

    for column in COLUMNS_TO_PLOT_TS:
        make_time_series_plot(df=covid19_df, column=column, start_date=START_DATE, countries_list=countries_list)

    for column in COLUMNS_TO_MODEL_EXPONENTIAL:
        model_exponential_and_forecast(df=covid19_df, column=column, start_date=START_DATE,
                                       days_ahead_to_forecast=DAYS_AHEAD_TO_FORECAST, countries_list=countries_list)

    for column in COLUMNS_TO_MODEL_POLYNOMIAL:
        fit_polynomial_regression_and_forecast(df=covid19_df, column=column, start_date=START_DATE,
                                               days_ahead_to_forecast=DAYS_AHEAD_TO_FORECAST,
                                               countries_list=countries_list)


if __name__ == "__main__":
    main()
