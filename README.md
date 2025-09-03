
# CIFAR10-CNN-PyTorch

En este proyecto se resolvera el dataset de CIFAR10 utilizando arquitecturas basadas en CNN.


Las **Convolutional Neural Networks** son un tipo de arquitectura de red neuronal que nacen del clasico **Multi Layer Perceptron**, pero que se diferencia por implementar capas de convolucion previas a las capas *fully connected*.

Las capas de convolucion se encargan basicamente de preprocesar los inputs de la red, para hacerlas mas nutritivas en su procesamiento.

La forma en la cual se logra preprocesar los inputs es utilizando justamente convolucion. La convolucion es el nombre que se le da a la operacion de aplicar un filtro sobre una matriz, siendo un filtro otra matriz de pesos aprendibles (en el contexto de los CNNs). La convolucion hace que se destaquen detalles relevantes de los inputs.

La magia de los CNNs (y la razon por la cual la convolucion se incluye dentro de la red) es que, dado que los filtros son matrices de pesos aprendibles, la red aprende a determinar que caracteristicas son mas destacables dentro de los inputs, ajustando los valores de los pesos dentro de los filtros usando backpropagation.


En este ejercicio en concreto, se utilizaran 3 arquitecturas basadas en CNN: plain CNN, Squeeze Excite Net (SENet) e InceptionNet. Luego se compararan sus resultados. Se espera que SENet sea con diferencia la mejor.

Ademas, se implementaran tecnicas de DataAugmentation y Feature Maps Visualization.