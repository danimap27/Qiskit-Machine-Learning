## 1. Introdución a las Redes Neuronales Cuánticas (QNN)
### 1.1 Redes Neuronales Cuánticas vs Redes Neuronales Clásicas

Las redes neuronales clásicas son modelos algorítmicos inspirados en el cerebro humano, que pueden entrenarse para reconocer patrones y resolver problemos complejos. Están compuestas por nodos interconectados, llamados neuronas, organizados en capas, y sus parámetros se ajustan mediante estrategias de aprendizaje automático o profundo.

El quantum machine learning (QML) busca combinar conceptos de computación cuántica y machine learning clásico para crear nuevos esquemas de aprendizaje. Las redes neuronales cuánticas (QNNs) son una integración entre redes neuronales clásicas y los circuitos cuánticos parametrizados. 

Las QNNs se pueden ser vistas desde dos perspectivas:

1. **Perspectiva de machine learning:** Las QNNs son modelos entrenables para detectar patrones en los datos, similares a las redes neuronales clásicas. Pueden cargar datos clásicos en un estado cuántico y procesarlos mediante puertas cuánticas paremetrizadas. Los resultados de estas mediciones se utilizan para entrenar los pesos mediante retroprogramación.
2. **Perpectiva de la computación cuántica:** Las QNNs son algoritmos cuánticos basados en circuitos cuánticos parametrizados. Estos circuitos incluyen un mapa de características (parámetros de entrada) y un ansatz (pesos ajustables).

<div style="text-align: center;">
<img src="images/img1.png" width="50%" height="50%" alt="Figura 1: Estructura genérica de una red neural cuántica">
</div>

Ambas perspectivas son complementarias, y no requieren definiciones estrictas de conceptos como "neuronas cuánticas" o las "capas" de una QNN.

### 1.2. Implementación en `qiskit-machine-learning`
En `qiskit-machine-learning`, las QNNs son unidades computacionales que se pueden adaptar a diferentes aplicaciones. Existen dos implementaciones principales:
* **[NeuralNetwork](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.NeuralNetwork.html):** Interfaz para redes neuronales, una clase abstracta de la cual heredan las QNNs.
* **[EstimatorQNN](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html):** Basada en la evaluación de observables mecánico-cuánticos.
* **[SamplerQNN](https://qiskit-community.github.io/qiskit-machine-learning/locale/fr_FR/stubs/qiskit_machine_learning.neural_networks.SamplerQNN.html):** Basada en muestras resultantes de la medición de un circuito cuántico.

Estas implementaciones se basan en los primitivos de Qiskit, que son el punto de entrada para ejecutar QNNs en simuladores o hardware cuántico real. Si no se proporciona una instancia del primitivo, la red crea una de forma automática.

Es importante destacar que las redes neuronales en qiskit-machine-learning no contienen capacidades de entrenamiento por sí solas, ya que estas se delegan a los algoritmos o aplicaciones específicos, como clasificadores o regresores.

```python
from qiskit_algorithms.utils import algorithm_globals

algorithm_globals.random_seed = 42
```

## 2. Cómo Instanciar QNNs
### 2.1 `EstimatorQNN`
`EstimatorQNN` toma como entrada un circuito cuántico parametrizado y un observable mecánico cuántico opcional para generar cálculos del valor esperado en el forward pass. Además, acepta listas de observables para construir QNNs más complejas.

**Ejemplo simple con `EstimatorQNN`:**
```python
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit

# Definir parámetros, input1 representa una entrada y weight1 un peso ajustable
params1 = [Parameter("input1"), Parameter("weight1")]

# Creamos un circuito cuántico
qc1 = QuantumCircuit(1)
# Añadimos las puertas cuánticas al circuito
# Puerta Hadamard al qubit 0
qc1.h(0)
# Rotación en el eje y al qubit 0, con un ángulo dado por input1
qc1.ry(params1[0], 0)
# Rotación en el eje y al qubit 0, con un ángulo dado por weigth1
qc1.rx(params1[1], 0)
# Dibuja el circuito
qc1.draw("mpl")
```
**Definición de un observable:**
```python
from qiskit.quantum_info import SparsePauliOp
# Definimos un observable usando la clase SparsePauliOp
# Este observable actúa sobre todos los qubits del circuito con la operación Pauli-Y
# El '1' es el coeficiente asociado con el término "Y", que representa la operación en los qubits
observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])
```

**Creación de `EstimatorQNN`:**
```python
from qiskit_machine_learning.neural_networks import EstimatorQNN
# Creamos una red neuronal cuántica estimadora (EstimatorQNN)
# 'circuit' es el circuito cuántico en el que se basa esta red neuronal
# 'observables' es el observable que queremos medir después de ejecutar el circuito
# 'input_params' y 'weight_params' son las variables de entrada y los parámetros de los pesos de la red neuronal
# El input_params define qué parte del circuito está influenciada por las entradas de datos y el weight_params representa los parámetros ajustables
estimator_qnn = EstimatorQNN(
    circuit=qc1, observables=observable1, input_params=[params1[0]], weight_params=[params1[1]]
)
```
### 2.2. `SamplerQNN`
`SamplerQNN` funciona similar a `EstimatorQNN`, pero consume muestras de medición directamente del circuito cuántico, sin necesidad de un observable personalizado.

**Ejemplo de `SamplerQNN`:**
```python
# 'SamplerQNN' es útil para modelos que no dependen de un observable específico para obtener información
from qiskit.circuit import ParameterVector

# Definimos dos conjuntos de parámetros: inputs2 (para las entradas) y weights2 (para los pesos del modelo)
inputs2 = ParameterVector("input", 2)
weights2 = ParameterVector("weight", 4)

# Creamos un circuito cuántico con 2 qubits
qc2 = QuantumCircuit(2)

# Aplicamos una serie de rotaciones Ry en ambos qubits, usando los parámetros de entrada
qc2.ry(inputs2[0], 0)
qc2.ry(inputs2[1], 1)

# Añadimos una puerta de CNOT entre los qubits 0 y 1, creando una correlación cuántica (entrelazamiento)
qc2.cx(0, 1)

# Aplicamos más rotaciones Ry en los qubits, esta vez usando los parámetros de los pesos
qc2.ry(weights2[0], 0)
qc2.ry(weights2[1], 1)

# Añadimos otra puerta CNOT para reforzar el entrelazamiento
qc2.cx(0, 1)

# Nuevas rotaciones Ry usando los últimos dos pesos
qc2.ry(weights2[2], 0)
qc2.ry(weights2[3], 1)

# Dibujamos el circuito
qc2.draw(output="mpl")
```
**Creación de `SamplerQNN`:**
```python
from qiskit_machine_learning.neural_networks import SamplerQNN

# Creamos una red neuronal cuántica basada en muestras de medición (SamplerQNN)
# Los parámetros de entrada son los que afectan a las entradas de datos en el circuito, y los parámetros de los pesos son ajustables durante el entrenamiento
sampler_qnn = SamplerQNN(circuit=qc2, input_params=inputs2, weight_params=weights2)
```

## 3. Como ejecutar Forward Pass
### 3.1. Configuración de entradas y pesos aleatorios
```python
from qiskit.utils import algorithm_globals

# Generamos valores aleatorios para las entradas y pesos de las redes neuronales cuánticas
# Esto es útil para inicializar los parámetros antes de entrenar el modelo
estimator_qnn_input = algorithm_globals.random.random(estimator_qnn.num_inputs)
estimator_qnn_weights = algorithm_globals.random.random(estimator_qnn.num_weights)

sampler_qnn_input = algorithm_globals.random.random(sampler_qnn.num_inputs)
sampler_qnn_weights = algorithm_globals.random.random(sampler_qnn.num_weights)
```
### 3.2. Forward Pass sin lotes