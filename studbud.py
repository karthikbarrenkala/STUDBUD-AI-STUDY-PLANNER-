import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import random

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Sample function to predict academic strength/weakness based on user input
def predict_strength_weakness(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Strength" if predicted_class == 1 else "Weakness"

# Define a function to create a study plan
def create_study_plan(goals, strengths, weaknesses, preferences):
    subjects = ['Math', 'Science', 'History', 'English', 'Programming']
    study_plan = []

    for subject in subjects:
        # Based on strengths and weaknesses, customize study hours
        if subject in strengths:
            study_hours = random.randint(1, 2)  # Less time for stronger subjects
        elif subject in weaknesses:
            study_hours = random.randint(3, 4)  # More time for weaker subjects
        else:
            study_hours = random.randint(2, 3)  # Average time for neutral subjects

        # Adjust based on the user's preferences for time of day
        if 'morning' in preferences:
            study_plan.append({'Subject': subject, 'Time': 'Morning', 'Hours': study_hours})
        elif 'afternoon' in preferences:
            study_plan.append({'Subject': subject, 'Time': 'Afternoon', 'Hours': study_hours})
        else:
            study_plan.append({'Subject': subject, 'Time': 'Evening', 'Hours': study_hours})

    return pd.DataFrame(study_plan)

# Sample Input Data (This would normally be gathered from user)
user_goals = ['Achieve better grades in Math', 'Improve programming skills']
user_strengths = ['Science', 'History']
user_weaknesses = ['Math', 'Programming']
user_preferences = ['morning', 'afternoon']

# Generate study plan
study_plan = create_study_plan(user_goals, user_strengths, user_weaknesses, user_preferences)

# Display the study plan
print("Your Personalized Study Plan:\n")
print(study_plan)
