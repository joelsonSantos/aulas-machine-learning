# Desvendando a Matemática da Regressão Linear: Do Cálculo à Solução Analítica

A Regressão Linear Simples é um dos algoritmos fundamentais em Machine Learning. Embora bibliotecas modernas abstraiam a matemática em uma única linha de código, compreender como o algoritmo otimiza seus parâmetros diferencia um engenheiro de software de um verdadeiro praticante de aprendizado de máquina. 

Este documento detalha o processo de derivação do método dos **Mínimos Quadrados Ordinários (MQO)**, partindo da função de custo até a equação de forma fechada.

---

## 1. O Problema: Minimizando Erros Quadráticos

O objetivo da regressão linear é encontrar a melhor reta que se ajusta a um conjunto de dados. Matematicamente, queremos encontrar o intercepto ($\beta_0$) e o coeficiente angular ($\beta_1$) que minimizem a distância entre os valores reais ($y_i$) e os valores previstos ($\hat{y}_i = \beta_0 + \beta_1x_i$).

Para isso, definimos a nossa função de custo como a **Soma dos Quadrados dos Resíduos (SQR)**:

$$SQR = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1x_i))^2$$

Elevamos o erro ao quadrado para garantir que desvios positivos e negativos não se anulem e para penalizar exponencialmente erros maiores.

---

## 2. O Cálculo: Derivadas Parciais e a Regra da Cadeia

Para encontrar o ponto de mínimo global dessa função convexa, aplicamos conceitos de cálculo multivariável. Precisamos calcular o gradiente da função de erro em relação a cada parâmetro e igualá-lo a zero. 

Aplicamos a **Regra da Cadeia** para derivar a expressão $u^2$, onde $u = (y_i - \beta_0 - \beta_1x_i)$.

### Derivada em relação ao intercepto ($\beta_0$)
A derivada do termo interno em relação a $\beta_0$ é $-1$. Multiplicando pela regra da potência:

$$\frac{\partial SQR}{\partial\beta_0} = \sum_{i=1}^{n} 2(y_i - \beta_0 - \beta_1x_i)(-1) = 0$$

### Derivada em relação ao coeficiente angular ($\beta_1$)
A derivada do termo interno em relação a $\beta_1$ é $-x_i$. Aplicando a regra:

$$\frac{\partial SQR}{\partial\beta_1} = \sum_{i=1}^{n} 2(y_i - \beta_0 - \beta_1x_i)(-x_i) = 0$$

Estas duas equações formam o sistema de otimização que precisamos resolver.

---

## 3. A Álgebra: Isolando os Coeficientes

Com o sistema montado, o próximo passo é manipulação algébrica para isolar $\beta_0$ e $\beta_1$.

### Passo 3.1: Encontrando a equação para $\beta_0$

Partindo da primeira derivada:
$$\sum_{i=1}^{n} -2(y_i - \beta_0 - \beta_1x_i) = 0$$

Dividimos por $-2$ e distribuímos o somatório:
$$\sum y_i - \sum \beta_0 - \sum \beta_1x_i = 0$$

Sabendo que somar uma constante $\beta_0$ por $n$ vezes resulta em $n\beta_0$:
$$\sum y_i - n\beta_0 - \beta_1\sum x_i = 0$$

Dividindo toda a equação pelo número de observações $n$:
$$\frac{\sum y_i}{n} - \beta_0 - \beta_1\frac{\sum x_i}{n} = 0$$

Substituindo pelas médias ($\bar{y}$ e $\bar{x}$), chegamos à equação do intercepto:
$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

### Passo 3.2: Encontrando a equação para $\beta_1$

Substituímos o valor de $\beta_0$ encontrado acima na segunda equação do nosso sistema:
$$\sum_{i=1}^{n} -2x_i(y_i - \beta_0 - \beta_1x_i) = 0$$

Dividimos por $-2$ e expandimos os termos:
$$\sum (x_iy_i - \beta_0x_i - \beta_1x_i^2) = 0$$

$$\sum x_iy_i - \beta_0\sum x_i - \beta_1\sum x_i^2 = 0$$

Substituímos $\beta_0$ por $(\bar{y} - \beta_1\bar{x})$:
$$\sum x_iy_i - (\bar{y} - \beta_1\bar{x})\sum x_i - \beta_1\sum x_i^2 = 0$$

Sabendo que $\sum x_i = n\bar{x}$, substituímos e agrupamos os termos com $\beta_1$:
$$\sum x_iy_i - n\bar{x}\bar{y} + \beta_1n\bar{x}^2 - \beta_1\sum x_i^2 = 0$$

$$\sum x_iy_i - n\bar{x}\bar{y} = \beta_1(\sum x_i^2 - n\bar{x}^2)$$

Isolando $\beta_1$, obtemos:
$$\beta_1 = \frac{\sum x_iy_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}$$

---

## 4. A Solução Analítica Final

A equação isolada para $\beta_1$ no passo anterior pode ser simplificada usando identidades estatísticas fundamentais. 

O numerador é a forma expandida da **Covariância** (não normalizada) entre $X$ e $Y$, enquanto o denominador é a **Variância** (não normalizada) de $X$. Isso nos leva à forma analítica elegante e definitiva para os coeficientes:

$$ \hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} = \frac{Covar(x,y)}{Var(x)} $$
$$ \hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x} $$

Estas são as equações exatas utilizadas por algoritmos estatísticos tradicionais para encontrar a reta de regressão em um único passe pelos dados, sem a necessidade de um processo iterativo.