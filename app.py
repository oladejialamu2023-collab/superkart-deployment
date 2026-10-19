from flask import Flask,jsonify,request
import io,joblib,pandas as pd
superkart_api=Flask(__name__)
model=joblib.load("superkart_model.joblib")
EXPECTED_COLUMNS=["Product_Weight","Product_Sugar_Content","Product_Allocated_Area","Product_MRP","Store_Size","Store_Location_City_Type","Store_Type","Product_Id_char","Store_Age_Years","Product_Type_Category"]
def validate(frame):
    missing=[c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    return frame[EXPECTED_COLUMNS]
@superkart_api.get("/health")
def health(): return jsonify({"status":"ok"}),200
@superkart_api.post("/v1/predict")
def predict():
    try:
        frame=validate(pd.DataFrame([request.get_json(force=True)]))
        return jsonify({"predicted_sales":round(float(model.predict(frame)[0]),2)}),200
    except Exception as exc: return jsonify({"error":str(exc)}),400
@superkart_api.post("/v1/predictbatch")
def predict_batch():
    try:
        frame=validate(pd.read_csv(io.BytesIO(request.files["file"].read())))
        predictions=model.predict(frame).round(2)
        return jsonify({str(i):float(v) for i,v in enumerate(predictions)}),200
    except Exception as exc: return jsonify({"error":str(exc)}),400
