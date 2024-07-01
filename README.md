
# Flappy Bird Environment + Deep Q-Learning 

Este proyecto implementa un entorno de Flappy Bird y un modelo de aprendizaje profundo utilizando Q-Learning para entrenar al agente a jugar.

## Descripción

El proyecto consta de tres scripts principales:

1. `FlappyBird_env.py`: Define el entorno de Flappy Bird.
2. `deep_q_learning.py`: Contiene la implementación del algoritmo Deep Q-Learning.
3. `test_NN.py`: Script para probar la red neuronal entrenada.

## Requisitos

Las librerías necesarias para ejecutar este proyecto son:
- Pygame
- Pynput
- NumPy
- Torch (PyTorch)
- Matplotlib

Puedes instalar las dependencias ejecutando:
```bash
pip install pygame pynput numpy torch matplotlib
```
## Descripción del Entorno
El entorno de Flappy Bird en este proyecto está diseñado para simular el juego original, con ciertas observaciones y acciones específicas que el agente puede realizar.

### Observaciones
El estado del entorno en cualquier momento dado se define por las siguientes observaciones:
- **Distancia del pájaro al tubo más próximo arriba:** Esta es la distancia vertical desde la posición del pájaro hasta la parte inferior del tubo más cercano situado por encima del pájaro.
- **Distancia del pájaro al tubo más próximo abajo:** Esta es la distancia vertical desde la posición del pájaro hasta la parte superior del tubo más cercano situado por debajo del pájaro.
- **Velocidad del pájaro:** Esta es la velocidad vertical actual del pájaro, que puede ser positiva (movimiento hacia abajo) o negativa (movimiento hacia arriba).

### Acciones
El agente puede realizar las siguientes acciones en el entorno:
- **No saltar (0):** El agente decide no hacer nada, y el pájaro sigue cayendo debido a la gravedad.
- **Saltar (1):** El agente decide que el pájaro debe saltar, lo que implica que el pájaro se mueve hacia arriba.

### Función de Recompensa
La función de recompensa está diseñada para guiar al agente hacia comportamientos deseables:
- **Buena posición (entre el espacio de la próxima tubería en eje Y):** +0.5 puntos
- **Entrar en el espacio entre dos tubos:** +1 punto
- **Colisionar con cualquier objeto o límites:** -1 punto
- **Todo lo demás:** 0 puntos

## Uso

### Jugar a Flappy Bird

Para jugar uno mismo, ejecuta:
```bash
python3 FlappyBird_env.py
```

Este script te permitirá jugar como si del juego normal se tratase.

### Entrenar el modelo

Para entrenar el modelo, ejecuta:
```bash
python3 deep_q_learning.py
```

Este script entrenará un agente usando el algoritmo Deep Q-Learning en el entorno de Flappy Bird definido.

### Probar el modelo

Para probar el modelo entrenado (recuerda añadir el "path" del modelo en la parte indicada en el código), ejecuta:
```bash
python3 test_NN.py
```

Este script cargará la red neuronal entrenada y la usará para jugar a Flappy Bird.

## Estructura del Proyecto

- `FlappyBird_env.py`: Define la lógica del juego y el entorno para el agente de aprendizaje. Utiliza Pygame para la visualización y Pynput para la interacción con el juego.
- `deep_q_learning.py`: Implementa el algoritmo Deep Q-Learning utilizando PyTorch, incluyendo la red neuronal, la lógica de entrenamiento, y la política de selección de acciones.
- `test_NN.py`: Prueba la red neuronal entrenada jugando el juego, cargando un modelo previamente entrenado y evaluando su desempeño en el entorno.

## Contacto

Para cualquier pregunta o sugerencia, no dudes en contactar.

---

© [2024] [Nallue]. Todos los derechos reservados.
