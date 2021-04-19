
from fastapi import FastAPI, Query
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('model')
brand_list = ["brand_Audi","brand_Citroën","brand_Fiat", "brand_Honda", "brand_Hyundai", "brand_Jaguar", "brand_Kia", "brand_Mazda", "brand_Mercedes", "brand_Mitsubishi", "brand_Nissan", "brand_Opel", "brand_Peugeot","brand_Porsche","brand_Renault","brand_Smart", "brand_Tesla", "brand_Toyota", "brand_VW ","brand_Volvo"]
app = FastAPI()
@app.get('/predict')
async def prediction(age:float, engine:float,mileage:float, brand:str = Query("brand_Audi",
                                       enum =brand_list)):
  index_brand = brand_list.index(brand)
  brand_array = [0] * len(brand_list)
  brand_array[index_brand] = 1
  input = [age, engine, mileage] + brand_array
  input = np.array(input)
  input = np.expand_dims(input, axis=0)
  prediction = model.predict(input)
  return round(float(prediction[0][0]),3)