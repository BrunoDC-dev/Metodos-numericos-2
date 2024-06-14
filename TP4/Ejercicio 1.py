import numpy as np

# Función de costo F(x) = (Ax - b)^T (Ax - b)
def cost_function(A, x, b):
    return np.linalg.norm(A @ x - b)**2

# Gradiente de la función de costo F
def gradient(A, x, b):
    return 2 * A.T @ (A @ x - b)

# Gradiente descendente
def gradient_descent(A, b, alpha, num_iterations, regularization=None, delta2=0):
    n, d = A.shape
    x = np.random.rand(d)
    cost_history = []

    for i in range(num_iterations):
        if regularization == 'L2':
            grad = gradient(A, x, b) + 2 * delta2 * x 
        else:
            grad = gradient(A, x, b)

        x = x - alpha * grad
        cost_history.append(cost_function(A, x, b))

    return x, cost_history

# Configuración del problema
n = 5
d = 100
A = np.random.rand(n, d)
b = np.random.rand(n)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
sigma_max = np.max(S)
lambda_max = np.max(np.linalg.eigvals(2 * A.T @ A)) #¿Chequeado que es es el Hessiano?
alpha = 1 / lambda_max
num_iterations = 1000
delta2 = 10**(-2) * sigma_max

# Minimización de F
x_min_F, cost_history_F = gradient_descent(A, b, alpha, num_iterations)

# Minimización de F2
x_min_F2, cost_history_F2 = gradient_descent(A, b, alpha, num_iterations, regularization='L2', delta2=delta2)

# Solución utilizando SVD
x_svd = np.linalg.pinv(A) @ b

# Análisis de resultados
print("Solución utilizando gradiente descendente para F:", x_min_F)
print("Solución utilizando gradiente descendente para F2:", x_min_F2)
print("Solución utilizando SVD:", x_svd)

# Graficar la evolución del costo
import matplotlib.pyplot as plt

plt.plot(cost_history_F, label='Costo F')
plt.plot(cost_history_F2, label='Costo F2')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.title('Evolución del costo')
plt.show()
