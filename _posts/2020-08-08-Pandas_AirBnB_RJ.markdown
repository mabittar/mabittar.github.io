---
layout: post
title: Apresentação da Biblioteca Pandas utilizando dados do AirBnB
date: 2020-08-08 00:00:00 +0300
description: Apresentação da biblioteca Pandas utilizando dados do AirBnB. # Add post description (optional)
img: rio.jpg # Add image post (optional)
tags: [Pandas, Data Analytics, Python] # add tag
---

Esse projeto consiste em apresentar as funcionalidades da biblioteca Pandas utilizando os dados do AirBnb disponíveis para a Cidade do Rio de Janeiro.

Neste [notebook](https://colab.research.google.com/drive/1IRgY4Ztu5x6kONevqYFL8nYiPelx6vB6?usp=sharing) irei apresentar os passos para importar uma base de dados diretamente para o Colab. A partir desses dados iremos utilizar ferramentas exploratórias como resumos estatísticos e histogramas.
Adiante com apenas uma linha de comando faremos um gráfico em forma de mapa de calor que mostra a correlação entre as variáveis (por exemplo reviews x preço).
Ao final serão apresentados exemplos de como variáveis não tratada prejudicam a análise do conjunto, assim como ferramentas para identificar e tratar esses desvios.

o post completo pode ser acessado [aqui](https://medium.com/@marcelmartinsbittar/apresenta%C3%A7%C3%A3o-da-biblioteca-pandas-a7b026bc33e5)


A seguir são apresentados alguns resultados obtidos com a biblioteca durante os estudos:

**Distribuição das Vairáveis**


    
    # plotar o histograma das variáveis numéricas
    df.hist(bins = 15, figsize=(15,10));
    
    

![](https://miro.medium.com/max/902/1*emYj1c_oK_7CG1SBygrtsg.jpeg)

**Correlação entre as Variáveis**
Para plotar um mapa de calor que indica visualmente a correlação entre as variáveis usamos

    
    sns.heatmap(corr, cmap='RdBu', fmt='.2f', square=True, linecolor='white', annot=True);
    

![](https://miro.medium.com/max/441/1*QusMgZWYmqDn9BglP-upZQ.png)