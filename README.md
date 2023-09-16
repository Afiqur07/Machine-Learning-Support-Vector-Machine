# Machine-Learning / Data Science -Support-Vector-Machine Project
Predicting Alzheimer's using effective implementation of Support Vector Machine (SVM) Machine Learning model

The Problem: The specific goal of this project is to develop a support vector machine (SVM) classifier that can predict if an individual belongs to the stable normal controls (sNC) group or the stable DAT (sDAT) group based on a high-dimensional glucose metabolism signature taken from several regions in the individual’s brain.

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/dd87cf8d-484e-434a-b8be-82238c52e768)

Methods Used: Cross-Validation (CV) based grid-search approach, Polynomial Kernel and Radial Basis Function (RBF) Kernel.

The Data: The “training” dataset consists of glucose metabolism
features taken from 14 cortical brain regions (see fdg pet.feature.info.txt for a list of these regions) across
237 sNC and 237 sDAT individuals given in train.fdg pet.sNC.csv and train.fdg pet.sDAT.csv respectively. The test.fdg pet.sNC.csv and test.fdg pet.sDAT.csv files correspond to a “test” dataset
with the same brain glucose metabolism features taken from another 415 sNC and 122 sDAT individuals respectively.

A snippet of one of the Data files:

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/44fd5be9-382e-46f6-89ff-780c5a684ed2)

Question 1 :

Train a linear SVM classification model to discriminate between the sNC and sDAT individuals based on the brain
glucose metabolism features. Use the “training” dataset and a cross-validation (CV) based grid-search approach to
tune the regularization parameter “C” of the linear SVM model. Using the “best” “C” setting, re-train on the entire
“training” dataset to obtain the final linear SVM classification model. Estimate the “Err” of this final model on the
“test” dataset. Plot and discuss the performance of the models explored during the “C” hyperparameter tuning phase
as function of “C”.

Answer: - 

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/9aa1bef4-417b-4e79-9f77-33988c78af30)

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/774bae55-413e-4448-addc-65880f943905)

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/1c2e9360-3428-4a58-9421-4319e8148768)

Discussion:

Plotting the regularisation parameter "C" as a function of the linear SVM classifier's performance.
The y-axis displays the mean cross-validated accuracy of the relevant model, while the x-axis
displays the values of C utilised in the hyperparameter tuning phase. The standard deviation of the
accuracy scores throughout the five folds of the cross-validation is shown by the dark region
surrounding the mean accuracy curve.
The graphic shows that, up until a certain point (about C=10), the mean cross-validated accuracy
rises as C grows. Following that, accuracy begins to decline when C is raised more. Since bigger
values of C have narrower margins and might lead to overfitting to the training data, this behaviour
is predicted. Therefore, the maximum accuracy score obtained during the hyperparameter tuning
phase, which in this case is C=10, is used to pick the ideal value of C that balances model
complexity and accuracy

Question 2: -

Now train a non-linear SVM classifier with a polynomial kernel for the sNC vs sDAT classification task. Use the
“training” dataset and a CV based grid-search approach to tune both the regularization parameter “C” as well as the
degree “d” parameter of the polynomial kernel. Using the “best” “(C, d)” setting, re-train on the entire “training”
dataset to obtain the final polynomial kernel SVM model. Estimate the “Err” of this final model on the “test” dataset.
Compare the performance of the polynomial kernel SVM model with that of the above obtained final linear SVM
model.

Answer: -

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/79708265-eb2b-467a-b786-b673e0c8446b)

Starting with the polynomial kernel SVM model, it was able to accurately classify 87% of the
samples with an accuracy of 0.87 on the test dataset. Additionally, it had a sensitivity of 0.95,
which meant that 95% of the people with stable DAT were accurately recognised. In contrast, the
model's specificity was 0.85, meaning it accurately recognised 85% of the people with stable
normal controls. The model's accuracy was 0.65, which indicates that it correctly identified
individuals as belonging to the sDAT group 65% of the time. Recall, sometimes referred to as the
true positive rate, was 0.95, meaning that 95% of the people with sDAT were properly identified
by the model. The model's balanced accuracy, which accounts for the imbalance between the two
classes and provides a more realistic view of its performance, was 0.90 at the end.
The accuracy of the linear SVM model, which was marginally inferior to that of the polynomial
kernel SVM model on the test dataset, was 0.85. The model's sensitivity, which was greater at 0.94
and showed that it accurately identified 94% of those with sDAT, was higher. The model's
specificity, 0.83, was a little bit lower than that of the polynomial kernel SVM model. The model's
accuracy was 0.62, which was less than the polynomial kernel SVM model's. The model's recall
and sensitivity both stood at 0.94. Finally, the model's balanced accuracy was 0.89, somewhat less
accurate than the polynomial kernel SVM model.
In conclusion, the polynomial kernel SVM model outperformed the linear SVM model in terms of
accuracy and balance, but lagged behind in terms of precision and somewhat lagged behind in
terms of specificity. It's crucial to remember that the exact objectives and limitations of the project
will ultimately determine which model is used, and that further hyperparameter tuning or testing
out several models may result in even greater performance.

Question 3: -

Repeat the above experiment by replacing the polynomial kernel with a radial basis function (RBF) kernel. Note that
here the RBF kernel parameter “γ” needs to be tuned instead of the “d” parameter of the polynomial kernel. Compare
the performance of the final RBF kernel SVM classification model with that of both the final polynomial kernel SVM
and the linear SVM models.

Answer: -

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/38500d32-4a4b-41a4-98e4-2f19d2960b15)

![image](https://github.com/Afiqur07/Machine-Learning-Support-Vector-Machine/assets/27920239/ed436c33-34f1-4d55-a020-dd3465493e7d)

The accuracy of the RBF kernel SVM model is 0.8734, which is more than the accuracy of the
linear SVM model (0.8547) and the polynomial kernel SVM model (0.87). In addition, the RBF
kernel SVM outperforms the other two models in terms of balanced accuracy (0.8978) and
sensitivity (0.9426), suggesting that it is more accurate at classifying sDAT people.
The RBF kernel SVM, however, has a larger false positive rate since its accuracy is lower (0.6534)
than that of the polynomial kernel SVM (0.65). The RBF kernel SVM's specificity (0.8530) is
likewise somewhat lower than the polynomial kernel SVM's (0.85), suggesting a greater
probability of false negatives.
In terms of accuracy, sensitivity, and balanced accuracy, the RBF kernel SVM outperforms the
polynomial and linear SVM models overall, although it has a little worse precision and specificity.
As a result, the optimum model to choose will rely on the precise specifications of the
categorization problem.
