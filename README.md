# Predict Machine Failure

## Run FastAPI
Clone repository    
``git clone https://github.com/Joanna-Khek/predict-machine-failure/``

cd into directory    
``cd predict-machine-failure``

Build Docker Image    
``docker build -t predict-machine-failure .``

Run Docker Container    
``docker run --rm -p 80:80 predict-machine-failure``

Access API to get prediction   
``http://localhost/docs``

Using CURL to get prediction   
``
curl -X 'POST' \
  'http://localhost/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "ProductID": "ABCDE",
  "Type": "M",
  "AirTemperature": 320.4,
  "ProcessTemperature": 302.5,
  "RotationalSpeed": 1500,
  "Torque": 36.0 ,
  "ToolWear": 150,
  "TWF": 0,
  "HDF": 0,
  "PWF": 0,
  "OSF": 0,
  "RNF": 0
}'
``
