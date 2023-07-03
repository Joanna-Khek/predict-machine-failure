# Predict Machine Failure

## Run FastAPI
Clone repository    
``git clone https://github.com/Joanna-Khek/predict-machine-failure/``

Build Docker Image    
``docker build -t predict-machine-failure .``

Run Docker Container    
``docker run --rm -p 80:80 predict-machine-failure``

Access API to get prediction   
``http://localhost/docs``
