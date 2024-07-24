import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('SSH.csv')
print(df.isnull().sum())

#características categóricas
df['user'] = df['user'].astype('category')
df = pd.get_dummies(df, columns=['user'])

# Exploración de datos
print(df.head())
print(df.describe())

# Análisis de características
plt.figure(figsize=(15, 10))
df.drop('class', axis=1).hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribución de características")
plt.show()

# Escalar las características numéricas
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('class', axis=1))
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #clasificador SVM
# svm_classifier = SVC(kernel='rbf')
# svm_classifier.fit(X_train, y_train)
# y_pred = svm_classifier.predict(X_test)


# Definir los hiperparámetros para la búsqueda en cuadrícula
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}
svc_classifier = SVC()
grid_search = GridSearchCV(svc_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros encontrados
best_params = grid_search.best_params_
print("Mejores hiperparámetros:", best_params)
best_svc_classifier = SVC(**best_params)
best_svc_classifier.fit(X_train, y_train)

# Predecir las etiquetas para los datos de prueba
y_pred = best_svc_classifier.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
