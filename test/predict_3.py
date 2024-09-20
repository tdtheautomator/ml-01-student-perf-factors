import pandas as pd
from src.pipeline.pipeline_prediction import PredictPipeline

sample_df = pd.DataFrame(
{"Hours_Studied": ['30'],
"Attendance": ['80'],
"Parental_Involvement": ['High'],
"Access_to_Resources": ['Medium'],
"Extracurricular_Activities": ['No'],
"Sleep_Hours": ['8'],
"Previous_Scores": ['80'],
"Motivation_Level": ['Low'],
"Internet_Access": ['Yes'],
"Tutoring_Sessions": ['6'],
"Family_Income": ['High'],
"Teacher_Quality": ['Low'],
"School_Type": ['Private'],
"Peer_Influence": ['Positive'],
"Physical_Activity": ['2'],
"Learning_Disabilities": ['Yes'],
"Parental_Education_Level": ['Postgraduate'],
"Distance_from_Home": ['Moderate'],
"Gender": ['Female']}
)

call_pipeline = PredictPipeline()
sample_result = call_pipeline.predict(sample_df)
sample_result = int(sample_result[0])
print(f"predicted exam score  {sample_result}")