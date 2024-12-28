import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import nest_asyncio
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Load the TensorFlow Lite model
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "harLSTM_model.tflite")

# Load TensorFlow Lite Interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the input model
class TimeSeriesInput(BaseModel):
    data: List[List[float]]

@app.post("/predict/")
async def predict(timeseries: TimeSeriesInput):
    try:
        # Validate tensor dimensions
        if len(timeseries.data) != 200 or any(len(row) != 9 for row in timeseries.data):
            raise HTTPException(status_code=400, detail="La dimensione del tensore deve essere 200x9")
        
        # Convert input to numpy array
        input_array = np.array(timeseries.data, dtype=np.float32)
        input_array = input_array.reshape((1, 200, 9))  # Reshape for the model
        
        # Set tensor and invoke the interpreter
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()

        # Get prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)
        
        return {"predicted_class": int(predicted_class[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/helloworld/")
async def hello_world():
    return {"message": "Hello, world!"}

# Apply asyncio patching
nest_asyncio.apply()

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
