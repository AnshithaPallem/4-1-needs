# nndl mid 2
# unit 3b
### Parameter Norm Penalties (Simplified)

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

