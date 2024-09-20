# Machine Learning Prediction of Student Performance 

Sample dataset from [Kaggle](https://www.kaggle.com/)<br />
A simple model using several categories to predict student's score.<br />

- Hours_Studied
- Attendance
- Parental_Involvement
- Access_to_Resources
- Extracurricular_Activities
- Sleep_Hours
- Previous_Scores
- Motivation_Level
- Internet_Access
- Tutoring_Sessions
- Family_Income
- Teacher_Quality
- School_Type
- Peer_Influence
- Physical_Activity
- Learning_Disabilities
- Parental_Education_Level
- Distance_from_Home
- Gender
- Exam_Score **(to be predicted)**

## Models Used

- Linear Regression
- CatBoosting Regressor
- Random Forest

## Note
Accuracy evaluation depends on system config and performance. <br />
This is a sample code to understand how it's built. <br />
The dataset is only about 6000 rows which not sufficient to train the model for accuracy and needs fine tuning.<br />
Models are **not** configured with any hypertuning parameters.

## Usage (CLI)

- Ensure anaconda is installed [Anaconda Download](https://www.anaconda.com/download)
- Clone git repo
- Create new virtual environment
```
conda create -p venv python==3.12
```
- Activate new virtual environment
```
conda activate ./venv
```
- Deploy requirements
```
pip install -r requirements.txt
```
- Start data ingestion, transformation and training
```
python .\src\components\data_ingestion.py
```
- Predict with sample data (feel free to try with other datasets), refer : .\test\readme.txt
```
python ./test/predict_1.py
python ./test/predict_2.py
python ./test/predict_3.py
```
- Deactivate new virtual environment
```
conda deactivate
```
- Detailed logs are generated at .\logs

## Usage (Jupyter Notebooks)

- Select kernel (virtual environment)
- For Exploratory Data Analysis (EDA) use .\notebooks\eda_student_performance.ipynb
- For Training use .\notebooks\train_student_performance.ipynb

## Known Issues

- Model times out or doesn't provide enough accuracy if system is busy.
- ModuleNotFoundError: No module named 'src' : ensure -e . is not commented in requirements.txt
- Deprecation warning for setup tools