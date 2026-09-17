# ============================================================
# AGROMITRA - Decision Engine
# ============================================================
# Converts AI disease prediction + environmental conditions into:
#   1. Confidence level
#   2. Environmental risk
#   3. Severity
#   4. Recommended action
# ============================================================

# ------------------------------------------------------------
# 1. CONFIDENCE LEVEL
# ------------------------------------------------------------
def get_confidence_level(confidence):
    """
    Classifies AI confidence into HIGH, MEDIUM or LOW.
    confidence should be between 0 and 1 (e.g., 0.95 = 95%).
    """
    if confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.60:
        return "MEDIUM"
    else:
        return "LOW"

# ------------------------------------------------------------
# 2. ENVIRONMENTAL RISK
# ------------------------------------------------------------
def get_environmental_risk(temperature, humidity, soil_moisture):
    """
    Calculates a simple environmental risk score based on sensors.
    Returns: risk_level (LOW / MEDIUM / HIGH), risk_score (int)
    """
    risk_score = 0

    # Temperature condition
    if temperature >= 28:
        risk_score += 1
    if temperature >= 32:
        risk_score += 1

    # Humidity condition
    if humidity >= 75:
        risk_score += 1
    if humidity >= 85:
        risk_score += 1

    # Soil moisture condition
    if soil_moisture >= 70:
        risk_score += 1
    if soil_moisture >= 85:
        risk_score += 1

    # Convert score into risk level
    if risk_score >= 4:
        risk_level = "HIGH"
    elif risk_score >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_level, risk_score

# ------------------------------------------------------------
# 3. SEVERITY
# ------------------------------------------------------------
def calculate_severity(disease, confidence, environmental_risk):
    """
    Estimates decision severity using AI confidence and environmental risk.
    """
    confidence_level = get_confidence_level(confidence)

    # Healthy crop check
    if disease == "Healthy":
        if confidence_level == "HIGH":
            return "NONE"
        elif confidence_level == "MEDIUM":
            return "LOW"
        else:
            return "UNCERTAIN"

    # Low-confidence disease prediction check
    if confidence_level == "LOW":
        return "UNCERTAIN"

    # High confidence + high environmental risk
    if confidence_level == "HIGH" and environmental_risk == "HIGH":
        return "HIGH"

    # Medium/high combinations
    if confidence_level == "HIGH" and environmental_risk == "MEDIUM":
        return "MODERATE"
    if confidence_level == "MEDIUM" and environmental_risk == "HIGH":
        return "MODERATE"
    if confidence_level == "MEDIUM" and environmental_risk == "MEDIUM":
        return "MODERATE"

    return "LOW"

# ------------------------------------------------------------
# 4. RECOMMENDATION
# ------------------------------------------------------------
def get_recommendation(disease, severity):
    """
    Generates action recommendations based on disease + severity.
    """
    if disease == "Healthy":
        if severity == "NONE":
            return "Crop appears healthy. Continue regular monitoring and normal crop care."
        elif severity == "LOW":
            return "Crop appears mostly healthy. Continue monitoring the plant for visible symptoms."
        else:
            return "The result is uncertain. Capture another clear leaf image and inspect the plant manually."

    if disease == "Early_Blight":
        if severity == "HIGH":
            return "Possible Early Blight with high environmental risk. Inspect affected plants immediately, remove severely affected leaves, avoid prolonged leaf wetness, and follow local crop management guidelines."
        elif severity == "MODERATE":
            return "Possible Early Blight. Inspect affected leaves, improve airflow around plants, and avoid unnecessary leaf wetness."
        elif severity == "LOW":
            return "Possible Early Blight at low decision severity. Inspect the affected leaf and continue regular monitoring."
        else:
            return "Early Blight prediction is uncertain. Capture another clear image and manually inspect the plant."

    if disease == "Late_Blight":
        if severity == "HIGH":
            return "Possible Late Blight with high environmental risk. Inspect affected plants immediately, isolate or remove severely affected material, reduce leaf wetness, and follow local disease-management practices."
        elif severity == "MODERATE":
            return "Possible Late Blight. Inspect affected leaves carefully, reduce prolonged leaf wetness, and improve airflow."
        elif severity == "LOW":
            return "Possible Late Blight at low decision severity. Inspect the affected leaf and continue close monitoring."
        else:
            return "Late Blight prediction is uncertain. Capture another clear image and manually inspect the plant."

    return "The detected condition is not recognized. Capture another clear image and inspect the plant manually."

# ------------------------------------------------------------
# 5. MAIN DECISION FUNCTION
# ------------------------------------------------------------
def make_decision(disease, confidence, temperature, humidity, soil_moisture):
    """
    Main AGROMITRA decision function combining AI + Sensors.
    """
    confidence = max(0.0, min(1.0, confidence))
    confidence_level = get_confidence_level(confidence)
    
    environmental_risk, risk_score = get_environmental_risk(temperature, humidity, soil_moisture)
    severity = calculate_severity(disease, confidence, environmental_risk)
    recommendation = get_recommendation(disease, severity)

    return {
        "disease": disease,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "confidence_level": confidence_level,
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "environmental_risk": environmental_risk,
        "environmental_risk_score": risk_score,
        "severity": severity,
        "recommendation": recommendation
    }

# ------------------------------------------------------------
# 6. TESTING SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("AGROMITRA DECISION ENGINE TEST")
    print("=" * 60)

    # Simulated test values
    disease = "Early_Blight"
    confidence = 0.9516
    temperature = 29
    humidity = 81
    soil_moisture = 34

    result = make_decision(disease, confidence, temperature, humidity, soil_moisture)

    print("\nAI RESULT")
    print("-" * 60)
    print("Disease           :", result["disease"])
    print("Confidence        :", result["confidence_percentage"], "%")
    print("Confidence Level  :", result["confidence_level"])

    print("\nENVIRONMENT")
    print("-" * 60)
    print("Temperature       :", result["temperature"], "°C")
    print("Humidity          :", result["humidity"], "%")
    print("Soil Moisture     :", result["soil_moisture"], "%")
    print("Environmental Risk:", result["environmental_risk"])
    print("Risk Score        :", result["environmental_risk_score"])

    print("\nFINAL DECISION")
    print("-" * 60)
    print("Severity          :", result["severity"])
    print("Recommendation    :", result["recommendation"])
    print("=" * 60)