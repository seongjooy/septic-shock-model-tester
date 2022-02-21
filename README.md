# Model Tester for Septic Shock Prediction
This project is to facilitate the testing of the model generated at: https://github.com/seongjooy/septic-shock-predictor. </br>
There are 5 hours of data available (excluding the current hour) to use to predict the patient's condition (shock onset) in the last hour.
In other words, data from the past 1, 2, 3, 4, 5 hours (all) can be used, but using data from only the past 1, 2 hours is also an option. This should differ in performance, but may also be valuable to test it out in terms of its tradeoff with speed or efficiency.

</br>
The python file asks for input on which hours to use to predict the onset of septic shock; user input will consist of the numbers that they wish to utilize to train the model. The model will then be trained based on those hours, and present results accordingly.
