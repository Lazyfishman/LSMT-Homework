
# LSTM Assignment

## Overview

This project is part of our coursework as business school students. The goal of this assignment is to understand and replicate the LSTM (Long Short-Term Memory) baseline model provided by our instructor. This work includes implementing the given baseline, addressing challenges during implementation, and analyzing its performance.

## Project Structure

The project is divided into two main parts:

1. **Baseline Replication**

   - Reproducing the LSTM baseline as instructed.
   - Understanding and implementing the key components of the model.
   - During the training process, we discovered that the original `main` function provided by the instructor could not run on our local machines. To address this, we modified the `main` function to ensure compatibility, while maintaining the same environment as specified in the baseline.

2. **Weight Conversion**

   - After training the model, we converted the trained weights into the required format as specified in the assignment.

## Results

Since no modifications were made to the model itself and we used the same environment as in the baseline, our results were identical to those provided by the instructor. The F1 score achieved was **0.981**.
However, we slightly optimized the model, reducing the loss from the baseline value of 0.07 to 0.053.
## Additional Attempts

We attempted to extend the project by modifying the output layer to test the model’s performance in predicting stock prices. To evaluate the model, we used residual analysis as a method to assess its accuracy and reliability.

~~However, due to limited time and our own technical constraints, we encountered persistent encoding issues during the training process~~As a result, this work remains incomplete. The code and related files for this additional attempt are included as an appendix in the `price_estimate` folder.

We used MAE, R², and RMSE as metrics instead of the F1 score to evaluate the model's performance in predicting the data.
As the result ,We later trained the model using 20 epochs. Although the MAE and RMSE metrics performed well, the R² metric consistently showed a low fit, indicating that the model is not suitable for predicting future price.
## Future
In the future, we plan to further explore the mechanisms of the LSTM model, study other research experiences in the field of economics, and improve our model.

