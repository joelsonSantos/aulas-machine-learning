import numpy as np

def linear_regression(x, y): 
    # Definir as médias
    mean_x, mean_y = np.mean(x), np.mean(y)
    
    # Calcular a covariância e variância
    S_xy = 0
    S_xx = 0
    
    #Laço para o somatório
    for i in range(0, len(x)):
        # termo covariância
        S_xy += (x[i] - mean_x)*(y[i] - mean_y)
        # termo variância
        S_xx += (x[i] - mean_x)**2
  
    # Calcular os coeficientes de regressão
    beta_1 = S_xy / S_xx
    beta_0 = mean_y - beta_1*mean_x
  
    # Retorna os coeficientes
    return beta_0, beta_1 

if __name__ == "__main__":
    # Exemplo de uso
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])
    
    beta_0, beta_1 = linear_regression(x, y)
    print(f"Coeficiente angular (beta_1): {beta_1}")
    print(f"Coeficiente linear (beta_0): {beta_0}")

    # previsão de novos valores
    x_new = np.array([6, 7, 8])
    # y' = B0 + B1 * x
    y_pred = beta_0 + beta_1 * x_new
    print(f"Previsão para novos valores de x: {y_pred}")