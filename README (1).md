# Predictive Analytics Platform


## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Landing Page](#landing-page)
- [Batch Prediction](#batch-prediction)
- [License](#license)

## Introduction
Welcome to the Predictive Analytics Platform Project! This project is designed to predict various outcomes using different machine learning models. It includes predictions for lead conversion, intern placement, and intern attrition.

## Features
- **Lead Conversion Prediction**: Predicts whether a lead will convert based on various features.
- **Intern Placement Prediction**: Predicts whether an intern will be placed based on their performance and other factors.
- **Intern Attrition Prediction**: Predicts whether an intern will leave the organization based on various attributes.
- **Dark Mode Toggle**: Users can switch between light and dark modes for better visual comfort.
- **Batch Prediction**: Users can upload CSV files for batch predictions.

## Installation
To get started with the Combined Project, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/dev-Ar5haD/prediction_project.git
   cd prediction_project
   ```

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the project**:
   ```sh
   python app.py
   ```

## Usage
After installation, you can use the Project by navigating to the appropriate URLs for each model. Here are some examples:

- **Lead Conversion Prediction**:
  - URL: `/model1`
  - Form fields: Age, Course, Lead Source, Total Visits, Last Activity, Country, City, Occupation, Tags, Lead Quality, Page Views Per Visit, Engagement Score, Qualification, Lead Interest Level, Days Since Last Interaction, Course Fee Offered, Potential Score, Log Time Spent on Website, Interaction Time-Hour

- **Intern Placement Prediction**:
  - URL: `/model2`
  - Form fields: Age, Department, Duration of Internship, Performance Score, Attendance Rate, Socioeconomic Status, Number of Completed Projects, Technical Skill Rating, Soft Skill Rating, Hours Worked per Week, Mentorship Level, Distance from Work, Recommendation Score

- **Intern Attrition Prediction**:
  - URL: `/model3`
  - Form fields: Age, Department, Duration of Internship, Attendance Rate, Socioeconomic Status, Participation in Projects, Hours Worked per Week, Mentorship Level, Distance from Work, Job Satisfaction, Work-Life Balance Index, Performance Adjusted Score, Socioeconomic-Performance Interaction

## Landing Page
The landing page provides a user-friendly interface to select the desired prediction model. It features a stylish design with animated snowflakes and a welcome screen. Users can choose from the following options:
- **Lead Conversion Prediction**: Navigate to the lead conversion prediction form.
- **Intern Placement Prediction**: Navigate to the intern placement prediction form.
- **Intern Attrition Prediction**: Navigate to the intern attrition prediction form.

## Batch Prediction
The Project supports batch predictions, allowing users to upload CSV files for bulk predictions. Here are the steps to use batch prediction:

1. **Lead Conversion Batch Prediction**:
   - URL: `/predict_batch_model1`
   - Upload a CSV file with the required fields for lead conversion prediction.

2. **Intern Placement Batch Prediction**:
   - URL: `/batch_predict_model2`
   - Upload a CSV file with the required fields for intern placement prediction.

3. **Intern Attrition Batch Prediction**:
   - URL: `/batch_predict`
   - Upload a CSV file with the required fields for intern attrition prediction.

The results will be returned as a downloadable CSV file with the predictions.

## License
This project is free of license. You are free to use, modify, and distribute this project without any restrictions.

---

*Thank You*
