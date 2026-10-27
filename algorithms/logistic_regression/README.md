# Desvendando a Matemática da Regressão Logística: Da Probabilidade ao Gradiente

Se a Regressão Linear Simples procura traçar uma linha contínua através dos dados, a Regressão Logística resolve um problema diferente: a **classificação binária**. O objetivo é traçar uma fronteira de decisão que separe duas classes (como 0 e 1, ou Verdadeiro e Falso) e prever a probabilidade de uma observação pertencer a uma delas.

Este documento detalha como passamos da combinação linear para a probabilidade, exploramos a função de custo (Entropia Cruzada) e derivamos o gradiente.

---

## 1. A Hipótese: A Função Sigmóide

Na regressão linear, a nossa previsão é $\hat{y} = \beta_0 + \beta_1x_1 + \dots + \beta_px_p$. No entanto, esta equação pode gerar valores infinitamente negativos ou positivos. Como queremos prever uma **probabilidade** (que deve estar estritamente entre 0 e 1), precisamos de uma função de mapeamento.

Introduzimos a **Função Sigmóide** ($\sigma$):

$$ \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} $$

Onde $z$ é a nossa regressão linear clássica ($z = \beta_0 + \beta_1x_1 + \dots$). A sigmóide "esmaga" qualquer valor de $z$ para um intervalo entre $0$ e $1$, formando uma curva em "S".

---

## 2. A Função de Custo: Entropia Cruzada Binária (Log Loss)

Na regressão linear, utilizámos a Soma dos Erros Quadráticos (MQO). Se aplicarmos o erro quadrático à função sigmóide, o resultado será uma função não-convexa (cheia de mínimos locais), o que impossibilita a optimização.

Para a Regressão Logística, recorremos à **Estimativa de Máxima Verossimilhança** para deduzir uma função de custo estritamente convexa, conhecida como *Binary Cross-Entropy* ou *Log Loss*:

$$ J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

*   Se a classe real for $y = 1$, a segunda parte da equação anula-se. O custo passa a ser $-\log(\hat{y})$. Quanto mais próxima de 1 for a previsão, mais próximo de 0 será o custo.
*   Se a classe real for $y = 0$, a primeira parte anula-se. O custo passa a ser $-\log(1 - \hat{y})$.

---

## 3. O Cálculo: A Regra da Cadeia e as Derivadas Parciais

Para encontrar os melhores coeficientes ($\beta_j$), precisamos de minimizar a função de custo $J(\beta)$. Para isso, calculamos as derivadas parciais (o gradiente) em relação a cada coeficiente. 

Dado que $\hat{y}$ é uma função dependente de $z$, que por sua vez é dependente de $\beta_j$, utilizamos a **Regra da Cadeia**:

$$ \frac{\partial J}{\partial \beta_j} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \beta_j} $$

Vamos derivar cada componente individualmente para uma única observação (para simplificar, omitimos temporariamente o somatório e a divisão por $n$):

### Passo 3.1: Derivada do Custo em relação à previsão ($\hat{y}$)
Aplicando a regra da derivada do logaritmo ($$\frac{d}{dx}\log(x) = \frac{1}{x}$$):
$$ \frac{\partial J}{\partial \hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right) = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} $$

### Passo 3.2: Derivada da Sigmóide ($\hat{y}$) em relação a $z$
A derivada da função sigmóide possui uma propriedade matemática elegante:
$$ \frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y}) $$

### Passo 3.3: Derivada de $z$ em relação ao coeficiente ($\beta_j$)
Dado que $$z = \beta_0x_0 + \beta_1x_1 + \dots + \beta_jx_j$$, a derivada em relação a um coeficiente específico $\beta_j$ é simplesmente a variável independente que o acompanha:
$$ \frac{\partial z}{\partial \beta_j} = x_j $$

*(Nota: para o intercepto $\beta_0$, convenciona-se que $x_0 = 1$)*

### Passo 3.4: Juntando tudo (O Produto das Derivadas)
Multiplicando os três componentes obtidos:
$$ \frac{\partial J}{\partial \beta_j} = \left( \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right) \cdot \Big( \hat{y}(1 - \hat{y}) \Big) \cdot x_j $$

Note como o denominador $$\hat{y}(1 - \hat{y})$$ corta perfeitamente com a derivada da sigmóide. Restituindo o somatório sobre as $n$ observações, chegamos à matriz do gradiente:

**$$ \frac{\partial J}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)x_{ij} $$**

---

## 4. O Paradoxo: Sem Solução Analítica

Note uma particularidade notável: **a equação final do gradiente da Regressão Logística é algebricamente idêntica à da Regressão Linear**. A única diferença é a natureza de $\hat{y}_i$ (que agora encapsula a função sigmóide não-linear).

No entanto, há uma diferença brutal na resolução. Na Regressão Linear, conseguimos igualar o gradiente a zero e isolar os $\beta$ algebraicamente, chegando a uma solução de forma fechada (as equações de covariância/variância).

Na Regressão Logística, **não é possível isolar os parâmetros $\beta$**. A não-linearidade da função exponencial dentro da sigmóide impede a obtenção de uma fórmula analítica direta. 

### A Solução: Optimização Iterativa
Para ajustar o modelo, somos obrigados a recorrer a métodos iterativos, como a **Descida do Gradiente (Gradient Descent)**. O algoritmo arranca com valores arbitrários para $\beta$ e atualiza-os gradualmente em direcção ao mínimo da função de custo:

$$ \beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j} $$
$$ \beta_j := \beta_j - \alpha \left[ \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)x_{ij} \right] $$

Onde $\alpha$ é a taxa de aprendizagem (*learning rate*). O algoritmo calcula o erro, multiplica-o pelo recurso $x$, e ajusta o peso passo a passo até o gradiente convergir para zero.
