import requests
import json

def get_valid_input(prompt, input_type=str, options=None):
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                print("This field is required. Please try again.")
                continue
                
            if input_type == int:
                value = int(value)
            elif input_type == float:
                value = float(value)
            elif input_type == str and options:
                value = value.capitalize()
                if value not in options:
                    print(f"Please enter one of: {', '.join(options)}")
                    continue
                    
            return value
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def get_user_input():
    print("\nPlease enter customer information:")
    
    # Numerical features
    input_data = {
        "SeniorCitizen": get_valid_input("SeniorCitizen (0 or 1): ", int, ["0", "1"]),
        "tenure": get_valid_input("Tenure (months): ", int),
        "MonthlyCharges": get_valid_input("Monthly Charges: ", float),
        "TotalCharges": get_valid_input("Total Charges: ", float)
    }
    
    # Categorical features with validation
    input_data["gender"] = get_valid_input(
        "Gender (Male/Female): ", 
        str, 
        ["Male", "Female"]
    )
    
    input_data["Partner"] = get_valid_input(
        "Partner (Yes/No): ",
        str,
        ["Yes", "No"]
    )
    
    input_data["Dependents"] = get_valid_input(
        "Dependents (Yes/No): ",
        str,
        ["Yes", "No"]
    )
    
    input_data["PhoneService"] = get_valid_input(
        "Phone Service (Yes/No): ",
        str,
        ["Yes", "No"]
    )
    
    input_data["MultipleLines"] = get_valid_input(
        "Multiple Lines (Yes/No/No phone service): ",
        str,
        ["Yes", "No", "No phone service"]
    )
    
    input_data["InternetService"] = get_valid_input(
        "Internet Service (DSL/Fiber optic/No): ",
        str,
        ["DSL", "Fiber optic", "No"]
    )
    
    # Set default values for internet-dependent services if no internet
    if input_data["InternetService"] == "No":
        no_internet = "No internet service"
        input_data.update({
            "OnlineSecurity": no_internet,
            "OnlineBackup": no_internet,
            "DeviceProtection": no_internet,
            "TechSupport": no_internet,
            "StreamingTV": no_internet,
            "StreamingMovies": no_internet
        })
    else:
        input_data["OnlineSecurity"] = get_valid_input(
            "Online Security (Yes/No/No internet service): ",
            str,
            ["Yes", "No", "No internet service"]
        )
        input_data["OnlineBackup"] = get_valid_input(
            "Online Backup (Yes/No/No internet service): ",
            str,
            ["Yes", "No", "No internet service"]
        )
        input_data["DeviceProtection"] = get_valid_input(
            "Device Protection (Yes/No/No internet service): ",
            str,
            ["Yes", "No", "No internet service"]
        )
        input_data["TechSupport"] = get_valid_input(
            "Tech Support (Yes/No/No internet service): ",
            str,
            ["Yes", "No", "No internet service"]
        )
        input_data["StreamingTV"] = get_valid_input(
            "Streaming TV (Yes/No/No internet service): ",
            str,
            ["Yes", "No", "No internet service"]
        )
        input_data["StreamingMovies"] = get_valid_input(
            "Streaming Movies (Yes/No/No internet service): ",
            str,
            ["Yes", "No", "No internet service"]
        )
    
    input_data["Contract"] = get_valid_input(
        "Contract (Month-to-month/One year/Two year): ",
        str,
        ["Month-to-month", "One year", "Two year"]
    )
    
    input_data["PaperlessBilling"] = get_valid_input(
        "Paperless Billing (Yes/No): ",
        str,
        ["Yes", "No"]
    )
    
    input_data["PaymentMethod"] = get_valid_input(
        "Payment Method (Bank transfer (automatic)/Credit card (automatic)/Electronic check/Mailed check): ",
        str,
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )
    
    return input_data

url = "http://localhost:8000/api/v1/predict"
headers = {'Content-Type': 'application/json'}

while True:
    print("\n=== Churn Prediction API Test ===")
    print("1. Enter new customer data")
    print("2. Exit")
    choice = input("\nPlease select an option (1-2): ")
    
    if choice == "1":
        try:
            input_data = get_user_input()
            shadow_mode = False
            
            print("\nSending prediction request...")
            response = requests.post(url, headers=headers, json={"input_data": input_data, "shadow_mode": shadow_mode})
            
            if response.status_code == 200:
                print("\nPrediction result:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\nError: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            
    elif choice == "2":
        print("\nGoodbye!")
        break
    else:
        print("\nInvalid choice. Please try again.")