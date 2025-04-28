import torch
import numpy as np
from model import CKDNet  # Import your model class

def predict_ckd(pcv, sc, hemo):
    # Load the saved model
    model = CKDNet()
    model.load_state_dict(torch.load('ckd_model.pth'))
    
    # Set the model to evaluation mode
    model.eval()
    
    # Prepare the input (similar to how we processed training data)
    features = np.array([pcv, sc, hemo], dtype=np.float32)
    features_tensor = torch.tensor(features).unsqueeze(0)  # Add batch dimension
    
    # Make prediction with no gradient calculation
    with torch.no_grad():
        output = model(features_tensor)
        
    # Convert to probability (0-1)
    probability = output.item()
    
    # Determine the prediction (ckd or no ckd)
    prediction = "ckd" if probability >= 0.5 else "no ckd"
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
    # Example values (replace with real patient data)
    patient_pcv = 40
    patient_sc = 1.1
    patient_hemo = 13.5
    
    result, confidence = predict_ckd(patient_pcv, patient_sc, patient_hemo)
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")