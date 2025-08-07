# ml_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NoShowPredictor:
    def __init__(self):
        # Load real dataset
        df = pd.read_csv("no_show_appointments.csv")  #CSV file

        # Preprocess: create the columns you want to use
        df['no_show'] = df['No-show'].map({'Yes': 1, 'No': 0})
        df['sms_reminder'] = df['SMS_received']
        df['days_between'] = (pd.to_datetime(df['AppointmentDay']) - pd.to_datetime(df['ScheduledDay'])).dt.days

        self.features = ['Age', 'sms_reminder', 'days_between']
        X = df[self.features]
        y = df['no_show']

        # Train model
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        importances = self.model.feature_importances_
        # plt.figure(figsize=(6, 4))
        # plt.bar(self.features, importances)
        # plt.title("Feature Importance - Random Forest")
        # plt.xlabel("Feature")
        # plt.ylabel("Importance Score")
        # plt.tight_layout()
        # plt.show()
        # print('\n')
        # df['no_show'].value_counts().plot(kind='bar')
        # plt.xticks([0, 1], ['Showed Up', 'No-Show'], rotation=0)
        # plt.title("Appointment Attendance")
        # plt.ylabel("Count")
        # plt.show()

    def predict(self, age, sms_reminder, days_between):
        df = pd.DataFrame([{
            'Age': age,
            'sms_reminder': sms_reminder,
            'days_between': days_between
        }])
        result = self.model.predict(df)
        return "Will No-Show" if result[0] == 1 else "Will Attend"
