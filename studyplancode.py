import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import random

# Load pre-trained BERT tokenizer and model only once to improve performance
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Function to predict academic strength/weakness based on user input
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

    # Simplify time allocation based on strength and weaknesses
    time_allocation = {'strength': (1, 2), 'neutral': (2, 3), 'weakness': (3, 4)}
    
    for subject in subjects:
        # Determine study time based on strengths and weaknesses
        if subject in strengths:
            study_hours = random.randint(*time_allocation['strength'])
        elif subject in weaknesses:
            study_hours = random.randint(*time_allocation['weakness'])
        else:
            study_hours = random.randint(*time_allocation['neutral'])

        # Select study time based on preferences
        for preference in preferences:
            if preference in ['morning', 'afternoon', 'evening']:
                study_plan.append({'Subject': subject, 'Time': preference.capitalize(), 'Hours': study_hours})
                break  # Stop after the first valid preference is applied

    return pd.DataFrame(study_plan)

# Sample Input Data
user_goals = ['Achieve better grades in Math', 'Improve programming skills']
user_strengths = ['Science', 'History']
user_weaknesses = ['Math', 'Programming']
user_preferences = ['morning', 'afternoon']  # Can specify more preferences

# Generate and display study plan
study_plan = create_study_plan(user_goals, user_strengths, user_weaknesses, user_preferences)
print("Your Personalized Study Plan:\n")
print(study_plan)
