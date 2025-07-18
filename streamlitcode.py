import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import random
import streamlit as st

# Load pre-trained BERT tokenizer and model
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

# Function to create a study plan
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

# Streamlit UI
st.title("Personalized Study Plan Generator")

# Input fields for user goals, strengths, weaknesses, and preferences
user_goals = st.text_area("Enter your goals (comma separated)", "Achieve better grades in Math, Improve programming skills").split(',')
user_strengths = st.text_area("Enter your strengths (comma separated)", "Science, History").split(',')
user_weaknesses = st.text_area("Enter your weaknesses (comma separated)", "Math, Programming").split(',')
user_preferences = st.multiselect("Select your preferred study times", ["Morning", "Afternoon", "Evening"])

# Generate study plan on button click
if st.button("Generate Study Plan"):
    # Generate the study plan based on user input
    study_plan = create_study_plan(user_goals, user_strengths, user_weaknesses, user_preferences)

    # Display the study plan
    st.write("### Your Personalized Study Plan:")
    st.dataframe(study_plan)

    # Optionally, you can display the predicted strengths/weaknesses for the goals
    for goal in user_goals:
        st.write(f"**Goal**: {goal}")
        prediction = predict_strength_weakness(goal)
        st.write(f"Predicted Strength/Weakness: {prediction}")

