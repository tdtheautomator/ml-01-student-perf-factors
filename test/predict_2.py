import pandas as pd
from src.pipeline.pipeline_prediction import PredictPipeline

sample_df = pd.DataFrame(
{"Hours_Studied": ['8'],
"Attendance": ['60'],
"Parental_Involvement": ['High'],
"Access_to_Resources": ['High'],
"Extracurricular_Activities": ['Yes'],
"Sleep_Hours": ['6'],
"Previous_Scores": ['90'],
"Motivation_Level": ['High'],
"Internet_Access": ['No'],
"Tutoring_Sessions": ['1'],
"Family_Income": ['Medium'],
"Teacher_Quality": ['High'],
"School_Type": ['Private'],
"Peer_Influence": ['Neutral'],
"Physical_Activity": ['4'],
"Learning_Disabilities": ['Yes'],
"Parental_Education_Level": ['Postgraduate'],
"Distance_from_Home": ['Near'],
"Gender": ['Female']}
)

call_pipeline = PredictPipeline()
sample_result = call_pipeline.predict(sample_df)
sample_result = int(sample_result[0])
print(f"predicted exam score  {sample_result}")