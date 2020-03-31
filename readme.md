## COVID-19 ETL and Line Fitting
This repository includes a simple script, 
covid19_analysis.py, to pull COVID-19 data from a verified EU
website. After ingestion, the script cleans the data, 
fits simple statistical models, and produces
visualizations. 

The script can be run in a Python virtual environment:
<br>
$ python3 -m venv venv
<br>
$ source venv/bin/activate
<br>
$ pip3 install -r requirements.txt
<br> 
$ python3 covid19_analysis.py
<br>
$ deactivate

Notable features of the script include the following:
* Only fetches remote data if it has not already been fetched 
during a given day. Otherwise, a cached csv is used. 
The EU website only updates once per day, so if we have already
pulled the data on a given day, we have the most recent copy.
* Columns and countries analyzed can be easily toggled with 
global lists.
* Both exponential and polynomial models are fit. Over time, as
the underlying real function becomes more complex, these models
will both overfit and diverge from one another. 
The scikit-learn framework is used to fit the 
polynomial model, a more flexible fit than an exponential. 
The process does not follow
some best practices, such as cross validation, as the focus
is more on fitting a line rather than optimizing for 
generalization. 
* A boolean flag exists to trigger a worldwide analysis
in addition to a country-specific analysis. 
