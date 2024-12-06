# nndl mid 2
# unit 3b
# Parameter Norm Penalties (Simplified)

Parameter norm penalties are a way to **control the size of the weights** in a deep learning model to prevent overfitting. This is done by adding a penalty to the loss function during training.

![Screenshot 2024-12-06 164130](https://github.com/user-attachments/assets/1e0546fb-a5e8-4a30-b152-195f914a1681)


#### Types of Penalties:
1. **L1 Regularization (Lasso)**:
   - Adds the **absolute values of the weights**:
    ![Screenshot 2024-12-06 164149](https://github.com/user-attachments/assets/81638600-8458-4f86-a8e1-4eec7103ee79)

   - **Effect**: Makes some weights exactly zero (sparse model).

2. **L2 Regularization (Ridge)**:
   - Adds the **squared values of the weights**:
     ![Screenshot 2024-12-06 164155](https://github.com/user-attachments/assets/43cb0f99-64a1-4a52-b723-077f968308de)

   - **Effect**: Shrinks weights towards zero (smooth model).

#### Why Use It?
- Prevents overfitting by stopping weights from becoming too large.
- Helps the model generalize better to new data.

- Without penalties: The model memorizes exact values (overfits).
- With penalties: The model learns trends, not exact values (generalizes better). 

# Norm Penalties as Constrained Optimization

Norm penalties try to minimize the loss (error) of a model and keep the model's parameters (weights) under control. They achieve this by adding a penalty for large parameters to the loss function.

![Screenshot 2024-12-06 164607](https://github.com/user-attachments/assets/c36ddfc8-99f0-4196-b493-8096317c8f70)

![Screenshot 2024-12-06 164614](https://github.com/user-attachments/assets/0e377957-3738-4fdd-953b-9dc6dbffe21e)

**Key Idea**: Norm penalties (like L1 or L2) ensure the model doesn't use overly large parameters, helping prevent overfitting and improving generalization. The penalty or constraint limits how complex the model can become.

# Regularization and Under-Constrained Problems

**Under-constrained problems** occur when the model has **more parameters** (weights) than the number of training examples. This makes the model too flexible, allowing it to fit even noise in the data, leading to **overfitting**.

Regularization adds **constraints or penalties** to the training process to prevent the model from becoming overly complex. It helps the model focus on learning meaningful patterns instead of memorizing the training data.

#### **Why is Regularization Important for Under-Constrained Problems?**
1. **Controls Model Complexity**: Limits how much the parameters can grow, making the model simpler.
2. **Improves Generalization**: Helps the model perform better on unseen data by avoiding overfitting.
3. **Adds Stability**: Makes training more robust by reducing sensitivity to noise.

#### **Examples of Regularization Techniques**:
1. **L1/L2 Regularization**: Penalizes large weights to control their size.
2. **Dropout**: Randomly deactivates neurons during training to prevent reliance on specific features.
3. **Early Stopping**: Stops training when validation performance stops improving.

# Early Stopping

Early stopping is a **regularization technique** to prevent overfitting in machine learning models. It works by monitoring the model’s performance on a **validation set** during training and stopping the training process when performance stops improving.

1. **Train the Model**: Train as usual and measure both training and validation losses at each step (or epoch).
2. **Monitor Validation Performance**: 
   - If the validation loss stops decreasing (or accuracy stops improving), it’s a sign of overfitting.
3. **Stop Training**: End training when validation performance plateaus or starts getting worse.

- Training too long can cause the model to memorize the training data (overfit), which hurts its ability to generalize to new data.
- Stopping at the right point keeps the model simple and avoids overfitting.

### **Example:**
- If the model's validation loss improves for the first 10 epochs but worsens afterward, stop training after 10 epochs instead of continuing unnecessarily.
  
# Parameter Tying and Parameter Sharing
# Dropout

# unit 4
# The Convolution Operation
# Pooling
# Variants of the Basic Convolution Function
# Structured Outputs
# Data Types
# Recurrent Neural Networks

# unit 5
# Under complete Autoencoders
# Regularized Autoencoders
# Representational Power
# Layer Size and Depth
# Stochastic Encoders and Decoders
# Denoising Autoencoders
