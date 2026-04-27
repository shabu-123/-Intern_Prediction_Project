from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import io
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load Model 1 files (AdaBoost model)
model1 = joblib.load('models/model_adaboost.pkl')
scaler1 = joblib.load('models/scaler1.pkl')
course_encoder = joblib.load('models/course_encoder.pkl')
lead_source_encoder = joblib.load('models/lead_source_encoder.pkl')
last_activity_encoder = joblib.load('models/last_activity_encoder.pkl')
country_encoder = joblib.load('models/country_encoder.pkl')
city_encoder = joblib.load('models/city_encoder.pkl')
occupation_encoder = joblib.load('models/occupation_encoder.pkl')
tags_encoder = joblib.load('models/tags_encoder.pkl')
lead_quality_encoder = joblib.load('models/lead_quality_encoder.pkl')
qualification_encoder = joblib.load('models/qualification_encoder.pkl')
feature_names1 = joblib.load('models/feature_names1.pkl')

# Load Model 2 files (Placement model)
model2 = joblib.load('models/placement_model.joblib')
scaler2 = joblib.load('models/scaler2.joblib')
feature_names2 = joblib.load('models/feature_names2.joblib')

# Define categorical encodings for Model 2
department_encoding = {
    "Data Science": 0,
    "Finance": 1,
    "Human Resources": 2,
    "Marketing": 3,
    "Web Development": 4
}

socioeconomic_status_encoding = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

mentorship_level_encoding = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

# Load Model 3 files (Decision Tree model)
dt_model = joblib.load('models/decision_tree_model.joblib')
scaler3 = joblib.load('models/scaler.joblib')
le_dep = joblib.load('models/le_dep.joblib')
le_ment = joblib.load('models/le_ment.joblib')
le_soc = joblib.load('models/le_soc.joblib')
le_job = joblib.load('models/le_job.joblib')
le_part = joblib.load('models/le_part.joblib')

@app.route('/')
def home():
    return render_template('landing_page.html')

@app.route('/model1')
def model1_page():
    return render_template('index_model1.html')

@app.route('/model2')
def model2_page():
    return render_template('index_model2.html')

@app.route('/model3')
def model3_page():
    return render_template('index_model3.html')

@app.route('/predict_batch_model1', methods=['POST'])
def predict_batch_model1():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Read the uploaded CSV file
        input_data = pd.read_csv(file)
        
        # Process the input data
        input_data['Course'] = course_encoder.transform(input_data['Course'])
        input_data['Lead Source'] = lead_source_encoder.transform(input_data['Lead Source'])
        input_data['Last Activity'] = last_activity_encoder.transform(input_data['Last Activity'])
        input_data['Country'] = country_encoder.transform(input_data['Country'])
        input_data['City'] = city_encoder.transform(input_data['City'])
        input_data['Occupation'] = occupation_encoder.transform(input_data['Occupation'])
        input_data['Tags'] = tags_encoder.transform(input_data['Tags'])
        input_data['Lead Quality'] = lead_quality_encoder.transform(input_data['Lead Quality'])
        input_data['Qualification'] = qualification_encoder.transform(input_data['Qualification'])
        
        # Reorder the columns based on feature_names
        input_data = input_data[feature_names1]
        
        # Standardize the features
        input_data_scaled = scaler1.transform(input_data)
        
        # Make predictions
        predictions = model1.predict(input_data_scaled)
        
        # Add predictions to the DataFrame
        input_data['Prediction'] = predictions
        
        # Convert numeric fields back to human-readable form
        input_data['Course'] = course_encoder.inverse_transform(input_data['Course'])
        input_data['Lead Source'] = lead_source_encoder.inverse_transform(input_data['Lead Source'])
        input_data['Last Activity'] = last_activity_encoder.inverse_transform(input_data['Last Activity'])
        input_data['Country'] = country_encoder.inverse_transform(input_data['Country'])
        input_data['City'] = city_encoder.inverse_transform(input_data['City'])
        input_data['Occupation'] = occupation_encoder.inverse_transform(input_data['Occupation'])
        input_data['Tags'] = tags_encoder.inverse_transform(input_data['Tags'])
        input_data['Lead Quality'] = lead_quality_encoder.inverse_transform(input_data['Lead Quality'])
        input_data['Qualification'] = qualification_encoder.inverse_transform(input_data['Qualification'])
        
        # Convert DataFrame to CSV
        output = io.BytesIO()
        input_data.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, mimetype='text/csv', download_name='predictions.csv', as_attachment=True)

# Model 1 Prediction Route
@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    try:
        # Collect form data for Model 1
        data = {
            'Age': request.form['age'],
            'Course': course_encoder.transform([request.form['course']])[0],
            'Lead Source': lead_source_encoder.transform([request.form['lead_source']])[0],
            'Total Visits': float(request.form['total_visits']),
            'Last Activity': last_activity_encoder.transform([request.form['last_activity']])[0],
            'Country': country_encoder.transform([request.form['country']])[0],
            'City': city_encoder.transform([request.form['city']])[0],
            'Occupation': occupation_encoder.transform([request.form['occupation']])[0],
            'Tags': tags_encoder.transform([request.form['tags']])[0],
            'Lead Quality': lead_quality_encoder.transform([request.form['lead_quality']])[0],
            'Page Views Per Visit': float(request.form['page_views_per_visit']),
            'Engagement Score': float(request.form['engagement_score']),
            'Qualification': qualification_encoder.transform([request.form['qualification']])[0],
            'Lead Interest Level': int(request.form['lead_interest_level']),
            'Days Since Last Interaction': int(request.form['days_since_last_interaction']),
            'Course Fee Offered': float(request.form['course_fee_offered']),
            'Potential Score': float(request.form['potential_score']),
            'Log Time Spent on Website': float(request.form['log_time_spent_on_website']),
            'Interaction Time-Hour': float(request.form['interaction_time_hour']),
        }

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([data])

        # Reorder columns based on feature_names
        input_data = input_data[feature_names1]

        # Standardize the features
        input_data_scaled = scaler1.transform(input_data)

        # Make prediction
        prediction = model1.predict(input_data_scaled)

        # Convert prediction to readable output
        prediction_text = "Converted" if prediction[0] == 1 else "Not Converted"

        return render_template('result_model1.html', prediction_text=prediction_text)

    except Exception as e:
        return str(e)

# Model 2 Prediction Route
@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    try:
        # Collect form data for Model 2
        data = request.form
        single_data = {
            'Age': int(data['age']),
            'Department': department_encoding[data['department']],
            'Duration of Internship (months)': int(data['duration']),
            'Performance Score': float(data['performance']),
            'Attendance Rate': float(data['attendance']),
            'Socioeconomic Status': socioeconomic_status_encoding[data['socioeconomic_status']],
            'Number of Completed Projects': int(data['projects']),
            'Technical Skill Rating': float(data['technical']),
            'Soft Skill Rating': float(data['soft']),
            'Hours Worked per Week': int(data['hours']),
            'Mentorship Level': mentorship_level_encoding[data['mentorship']],
            'Distance from Work (miles)': float(data['distance']),
            'Recommendation Score': float(data['recommendation'])
        }

        # Convert to DataFrame
        single_data_df = pd.DataFrame([single_data])

        # Ensure correct column order
        single_data_df = single_data_df[feature_names2]

        # Scale numerical features
        numerical_features = [
            'Age', 'Duration of Internship (months)', 'Performance Score', 'Attendance Rate',
            'Number of Completed Projects', 'Technical Skill Rating', 'Soft Skill Rating',
            'Hours Worked per Week', 'Distance from Work (miles)', 'Recommendation Score'
        ]
        single_data_numerical = single_data_df[numerical_features]
        single_data_scaled = scaler2.transform(single_data_numerical)

        # Combine scaled and categorical features
        single_data_scaled_df = pd.DataFrame(single_data_scaled, columns=numerical_features, index=single_data_df.index)
        single_data_final = pd.concat([single_data_scaled_df, single_data_df[['Department', 'Socioeconomic Status', 'Mentorship Level']]], axis=1)
        single_data_final = single_data_final[feature_names2]

        # Make prediction
        single_prediction = model2.predict(single_data_final)[0]

        # Map prediction to labels and emojis
        predicted_label = 'Placed' if single_prediction == 1 else 'Not Placed'
        emoji = '😀' if single_prediction == 1 else '😞'

        return render_template('result_model2.html', prediction_text=predicted_label, emoji=emoji)

    except Exception as e:
        return render_template('result_model2.html', prediction_text=f"Error: {str(e)}")
            
@app.route('/batch_predict_model2', methods=['POST'])
def batch_predict_model2():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return render_template('result_model2.html', prediction_text="No file uploaded")

        # Read the CSV file into a DataFrame
        batch_data = pd.read_csv(file)

        # Encode categorical features
        batch_data['Department'] = batch_data['Department'].map(department_encoding)
        batch_data['Socioeconomic Status'] = batch_data['Socioeconomic Status'].map(socioeconomic_status_encoding)
        batch_data['Mentorship Level'] = batch_data['Mentorship Level'].map(mentorship_level_encoding)

        # Ensure correct column order
        batch_data = batch_data[feature_names2]

        # Scale numerical features
        numerical_features = [
            'Age', 'Duration of Internship (months)', 'Performance Score', 'Attendance Rate',
            'Number of Completed Projects', 'Technical Skill Rating', 'Soft Skill Rating',
            'Hours Worked per Week', 'Distance from Work (miles)', 'Recommendation Score'
        ]
        batch_data_numerical = batch_data[numerical_features]
        batch_data_scaled = scaler2.transform(batch_data_numerical)

        # Combine scaled and categorical features
        batch_data_scaled_df = pd.DataFrame(batch_data_scaled, columns=numerical_features, index=batch_data.index)
        batch_data_final = pd.concat([batch_data_scaled_df, batch_data[['Department', 'Socioeconomic Status', 'Mentorship Level']]], axis=1)
        batch_data_final = batch_data_final[feature_names2]

        # Make predictions
        batch_predictions = model2.predict(batch_data_final)

        # Add predictions to the DataFrame
        batch_data['Placement Prediction'] = batch_predictions
        batch_data['Placement Prediction'] = batch_data['Placement Prediction'].map({1: 'Placed', 0: 'Not Placed'})

        # Convert numeric fields back to human-readable form
        batch_data['Department'] = batch_data['Department'].map({v: k for k, v in department_encoding.items()})
        batch_data['Socioeconomic Status'] = batch_data['Socioeconomic Status'].map({v: k for k, v in socioeconomic_status_encoding.items()})
        batch_data['Mentorship Level'] = batch_data['Mentorship Level'].map({v: k for k, v in mentorship_level_encoding.items()})

        # Convert DataFrame to CSV
        output = io.StringIO()
        batch_data.to_csv(output, index=False)
        output.seek(0)

        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='batch_predictions.csv')

    except Exception as e:
        return render_template('result_model2.html', prediction_model2=f"Error: {str(e)}")

# Model 3 Prediction Route
@app.route('/predict', methods=['POST'])
def predict_model3():
    # Get data from the form
    age = int(request.form['age'])
    department = request.form['department']
    duration = int(request.form['duration'])
    attendance_rate = float(request.form['attendance_rate'])
    socioeconomic_status = request.form['socioeconomic_status']
    participation = request.form['participation']
    hours_worked = int(request.form['hours_worked'])
    mentorship_level = request.form['mentorship_level']
    distance_from_work = float(request.form['distance_from_work'])
    job_satisfaction = request.form['job_satisfaction']
    work_life_balance = float(request.form['work_life_balance'])
    performance_score = float(request.form['performance_score'])
    interaction_score = float(request.form['interaction_score'])

    # Encode categorical features
    dep_encoded = le_dep.transform([department])[0]
    soc_encoded = le_soc.transform([socioeconomic_status])[0]
    part_encoded = le_part.transform([participation])[0]
    ment_encoded = le_ment.transform([mentorship_level])[0]
    job_encoded = le_job.transform([job_satisfaction])[0]

    # Create input array for prediction
    input_data = np.array([[age, dep_encoded, duration, attendance_rate, soc_encoded,
                            part_encoded, hours_worked, ment_encoded, distance_from_work,
                            job_encoded, work_life_balance, performance_score, interaction_score]])

    # Scale the input data
    scaled_input = scaler3.transform(input_data)

    # Make prediction
    prediction = dt_model.predict(scaled_input)

    # Render the result page with the prediction
    return render_template('result_model3.html', prediction=prediction[0])

@app.route('/batch_predict', methods=['POST'])
def batch_predict_model3():
    # Get the uploaded file
    file = request.files['csv_file']
    if not file:
        return "No file"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Rename columns to match the expected column names
    df.columns = ['age', 'department', 'duration', 'attendance_rate', 'socioeconomic_status',
                  'participation', 'hours_worked', 'mentorship_level', 'distance_from_work',
                  'job_satisfaction', 'work_life_balance', 'performance_score', 'interaction_score']

    # Check for required columns
    required_columns = ['age', 'department', 'duration', 'attendance_rate', 'socioeconomic_status',
                        'participation', 'hours_worked', 'mentorship_level', 'distance_from_work',
                        'job_satisfaction', 'work_life_balance', 'performance_score', 'interaction_score']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return f"Missing required columns: {', '.join(missing_columns)}"

    # Encode categorical features
    df['department'] = le_dep.transform(df['department'])
    df['socioeconomic_status'] = le_soc.transform(df['socioeconomic_status'])
    df['participation'] = le_part.transform(df['participation'])
    df['mentorship_level'] = le_ment.transform(df['mentorship_level'])
    df['job_satisfaction'] = le_job.transform(df['job_satisfaction'])

    # Prepare input data for prediction
    input_data = df.values

    # Scale the input data
    scaled_input = scaler3.transform(input_data)

    # Make predictions
    predictions = dt_model.predict(scaled_input)

    # Add predictions to the DataFrame
    df['prediction'] = predictions

    # Convert numeric fields back to human-readable form
    df['department'] = le_dep.inverse_transform(df['department'])
    df['socioeconomic_status'] = le_soc.inverse_transform(df['socioeconomic_status'])
    df['participation'] = le_part.inverse_transform(df['participation'])
    df['mentorship_level'] = le_ment.inverse_transform(df['mentorship_level'])
    df['job_satisfaction'] = le_job.inverse_transform(df['job_satisfaction'])

    # Save the DataFrame to a CSV file
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Send the CSV file as a response
    return send_file(output, mimetype='text/csv', download_name='predictions.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
