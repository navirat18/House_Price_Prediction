import gradio as gr
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("house_price_dataset.csv")
data = data.dropna()

# Features and target
x = data.drop("price", axis=1)
y = data["price"]

# Train-test split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(xtrain, ytrain)

# Gradio Interface (inline prediction)
interface = gr.Interface(
    fn=lambda bed, bath, size, loc, age: f"Predicted House Price: {model.predict([[bed, bath, size, loc, age]])[0]:,.2f}",
    inputs=[
        gr.Number(label="Number of Bedrooms", value=3),
        gr.Number(label="Number of Bathrooms", value=2),
        gr.Number(label="Size (sq feet)", value=1200),
        gr.Slider(1, 10, step=1, label="Location (1-10)", value=5),
        gr.Number(label="Age of the House", value=5),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="House Price Prediction by ",
    description="Enter house details to predict price using XGBoost."
)

if __name__ == "__main__":
    interface.launch()