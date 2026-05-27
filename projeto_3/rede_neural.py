import numpy as np

from sklearn.datasets import make_moons

def loss(y_pred, y_real):
    """
    Função de Cross Entropy que irá calcular o erro da rede para cada batch
    """
    # Evita o erro de log(0) adicionando um valorzinho (epsilon)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    log_y_pred = np.log(y_pred)
    log_y_pred_complemento = np.log(1 - y_pred)
    L = -(y_real * log_y_pred + (1 - y_real) * log_y_pred_complemento)
    L_medio = np.mean(L)
    return L_medio

def ativacao_relu(comb_l):
    """
    Função ReLu, que escolhe o máximo entre 0 e o resultado de z
    """
    return np.maximum(0,comb_l)

def ativacao_softmax(vals):
    """
    Função de ativação Softmax, que fica na saída da rede neural.
    Transforma as saídas da última camada em probabilidades e entrega a maior probabilidade como saída.
    Sua saída são as probabilidades de cada elemento
    """
    soma_exp = np.sum(np.exp(vals), axis=1, keepdims=True)

    probs = np.exp(vals)/soma_exp
    return probs

def neuronio(x, w, b):
    """
    w é a matriz de pesos, x é a matriz de entradas e b é o vetor de bias
    """
    w_x = np.dot(x, w)
    z = w_x + b
    return z

def foward_prop(entrada, pesos, bias, a, z_lista):
    """
    O Foward Propagation realiza o sentido de ida da rede neural.

    Pesos inicialmente definidos e não serão alterados aqui dentro.

    Esta rede neural é fully conected!!

    input:

    entrada: nd.array nx2, onde n é o tamanho da entrada.

    pesos: nd.array ixjxn, onde i é a quantidade de camadas, j é a quantidade de neurônios por rede.

    bias: lista de nd.array no formato 1xm, onde m é o número de neurônios por camada

    saida: nd.array de saída do Foward Propagation. Representa a probabilidade do ponto ser de determinada categoria.
    """

    a_camada = entrada
    a.append(a_camada)

    # Aqui são feitas as contas das camadas escondidas
    for i in range(len(pesos)):
        if i < (len(pesos) - 1):  # Nas camadas escondidas ele utiliza a função de ativação relu
            z = neuronio(a_camada, pesos[i], bias[i])
            z_lista.append(z)
            a_camada = ativacao_relu(z)
            a.append(a_camada)
        else:   # Na última camada ele utiliza a função de ativação softmax e terá como matriz final o formato batchx2
            z = neuronio(a_camada, pesos[i], bias[i])
            z_lista.append(z)
            a_camada = ativacao_softmax(z)

    return a_camada

def back_prop(y_pred, y_real, lista_ativ, pesos, bias, alfa, batch, z):
    """
    Algoritmo de Back Propagation. Este algoritmo já altera os pesos.
    """
    delta = y_pred - y_real # Delta da última camada
    i = len(bias) - 1
    while i >= 0:
        dw = np.dot(lista_ativ[i].T, delta) / batch
        db = np.sum(delta, axis=0, keepdims=True) / batch   # Tira a média das colunas, que será para o viés
        
        peso_backup = pesos[i].copy()
        
        # Atualização dos pesos e do bias
        pesos[i] = pesos[i] - (alfa * dw)
        bias[i] = bias[i] - (alfa * db)
        
        i -= 1
        if i >= 0:
            derivada_relu = (z[i] > 0).astype(float)
            delta = np.dot(delta, peso_backup.T) * derivada_relu    # Delta da atual é o delta da anterior vezes o peso da anterior vezes a derivada da ativação da atual

def treino_teste(ent, lab, prop_treino, prop_teste):
    """
    Esta função separa aleatoriamente os dados e labels para treino e para teste
    """
    treino_tam = int(ent.shape[0] * prop_treino)
    teste_tam = int(lab.shape[0] * prop_teste)

    rng = np.random.default_rng()
    indices = rng.permutation(ent.shape[0])   # Lista de índices aleatórios de tamanho total

    indices_treino = indices[:treino_tam]    # Lista de índices aleatórios com o tamanho do treino
    indices_teste = indices[treino_tam:]     # Lista de índices aleatórios com o tamanho do teste

    return ent[indices_treino], ent[indices_teste], lab[indices_treino], lab[indices_teste]

# Hiperparâmetros
hiperparam = {'numero_samples': 500,
              'batch': 20,
              'numero_camadas': 2,
              'neuronios_camadas': 16,
              'epocas': 1000,
              'prop_treino': 4/5,
              'prop_teste': 1/5,
              'tam_saida': 2,
              'alpha': 0.1  # Taxa de aprendizado
              }

# Crio as entradas no formato nx2 das duas luas com seus respectivos labels no formato nx1
# Ambas estão em formato ndarray
entrada, labels = make_moons(n_samples=hiperparam['numero_samples'], noise=0.1, random_state=42)

# Transforma o labels em formato (nx2) igual ao da entrada
labels = labels.reshape(-1,1)
labels = np.tile(labels, (1,2))
labels[:,1] = 1 - labels[:,1]

lista_pesos = []    # Os pesos serão guardados numa lista de pesos
lista_bias = []     # Os bias serão guardados numa lista de bias

# Criação inicial dos pesos e dos bias
# O laço irá realizar uma operação a mais por conta da camada de saída
for j in range(hiperparam['numero_camadas'] + 1):
    rng = np.random.default_rng()

    if j == 0:  # Primeira camada escondida que transformará a entrada em formato (Batch)x(Nº de neurônios por camada)
        pesos = rng.normal(loc=0.0,
                           scale=np.sqrt(2/entrada.shape[1]),   # Me baseei na inicialização de Kaiming
                           size=(2, hiperparam['neuronios_camadas'])
                           )
        bias = np.zeros((1, hiperparam['neuronios_camadas']))
        lista_bias.append(bias)
        lista_pesos.append(pesos)
    elif (hiperparam['numero_camadas'] > 1) and j > 0 and j < (hiperparam['numero_camadas']):    # Camadas escondidas subsequentes
        pesos = rng.normal(loc=0.0,
                           scale=np.sqrt(2/hiperparam['neuronios_camadas']),   # Me baseei na inicialização de Kaiming
                           size=(hiperparam['neuronios_camadas'], hiperparam['neuronios_camadas'])
                           )
        bias = np.zeros((1, hiperparam['neuronios_camadas']))
        lista_bias.append(bias)
        lista_pesos.append(pesos)
    else:   # Última camada do modelo
        pesos = rng.normal(loc=0.0,
                           scale=np.sqrt(2/hiperparam['neuronios_camadas']),   # Me baseei na inicialização de Kaiming
                           size=(hiperparam['neuronios_camadas'], 2)
                           )
        bias = np.zeros((1, 2))
        lista_bias.append(bias)
        lista_pesos.append(pesos)

# Irá ser executado a cada época
# Os dados e labels serão separados aleatoriamente a cada época
for i in range(hiperparam['epocas']):

    # 1. Seleciono aleatoriamente os dados de entrada e de treino para cada época
    entrada_treino, entrada_teste, labels_treino, labels_teste = treino_teste(entrada, labels, hiperparam['prop_treino'], hiperparam['prop_teste'])
    
    if (len(entrada_treino) % hiperparam['batch']) == 0:
        rep = int(len(entrada_treino) / hiperparam['batch'])
        div = 1
    else:
        rep = int(len(entrada_treino) / hiperparam['batch']) + 1
        div = 0
    
    for j in range(rep):    # Laço que irá executar os batchs até acabar a época atual. Trata o caso de não ser divisível
        a_lista = [] # crio uma lista que vai guardar a saída de cada camada
        z_lista = []
        batch = hiperparam['batch']

        if (j == (rep - 1)) and div == 0:   # Só entra na última execução E quando não é divisível
            # 2. Foward propagation de uma época
            y_pred = foward_prop(entrada_treino[(j*batch):,:], lista_pesos, lista_bias, a_lista, z_lista)
            # 3. Cálculo da Loss usando cross-entropy
            L_medio = loss(y_pred, labels_treino[(j*batch):,:])

            # 4. Back propagation de uma época junto com o Gradient Descent
            back_prop(y_pred, labels_treino[(j*batch):,:], a_lista, lista_pesos, lista_bias, hiperparam['alpha'], batch, z_lista)
        else:
            # 2. Foward propagation de uma época
            y_pred = foward_prop(entrada_treino[j*batch:(j+1)*batch,:], lista_pesos, lista_bias, a_lista, z_lista)

            # 3. Cálculo da Loss média por batch usando cross-entropy
            L_medio = loss(y_pred, labels_treino[j*batch:(j+1)*batch,:])

            # 4. Back propagation de uma época junto com o Gradient Descent
            back_prop(y_pred, labels_treino[j*batch:(j+1)*batch,:], a_lista, lista_pesos, lista_bias, hiperparam['alpha'], batch, z_lista)
    
    # Valor do Loss do treino após o final da época
    print('treino:')
    print(L_medio)

    # Implementação do teste
    a_lista = []
    z_lista = []

    y_pred = foward_prop(entrada_teste, lista_pesos, lista_bias, a_lista, z_lista)

    L_medio = loss(y_pred, labels_teste)
    
    print('teste:')
    print(L_medio)
