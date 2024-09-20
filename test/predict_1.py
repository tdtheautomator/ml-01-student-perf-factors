import pandas as pd
from src.pipeline.pipeline_prediction import PredictPipeline

sample_df = pd.DataFrame(
{"Hours_Studied": ['10'],
"Attendance": ['50'],
"Parental_Involvement": ['Medium'],
"Access_to_Resources": ['Low'],
"Extracurricular_Activities": ['No'],
"Sleep_Hours": ['5'],
"Previous_Scores": ['40'],
"Motivation_Level": ['Medium'],
"Internet_Access": ['Yes'],
"Tutoring_Sessions": ['2'],
"Family_Income": ['Low'],
"Teacher_Quality": ['Medium'],
"School_Type": ['Public'],
"Peer_Influence": ['Negative'],
"Physical_Activity": ['3'],
"Learning_Disabilities": ['No'],
"Parental_Education_Level": ['High School'],
"Distance_from_Home": ['Near'],
"Gender": ['Male']}
)

call_pipeline = PredictPipeline()
sample_result = call_pipeline.predict(sample_df)
sample_result = int(sample_result[0])
print(f"predicted exam score  {sample_result}")