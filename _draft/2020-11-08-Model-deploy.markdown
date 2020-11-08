---
layout: post
title: Machine Learning Deploy 
date: 2020-11-08 00:00:00 +0300
description: Desenvolvimento e disponibilização de um modelo de Machine Learnin -  # Add post description (optional)
img: deploy.jpg # Add image post (optional)
tags: [Machine Learning, Deploy, Imsomnia, Heroku] # add tag

#sniptes activated: {%%} - highlight python; ![] - new images
---

## Desenvolvimento de um modelo de Machine Learning

Por que aproximadamente 90% dos modelos de Machine Learning não evoluem de um trabalho acadêmico ou de um notebook no Google Colab para um modelo online, para que possa alimentar outras aplicações?

Uma das respostas para a pergunta incial é que a maioria das orgnizações ainda não estão familiarizadas com a tecnologia e ferramentas similares, muito menos possuem hardware necessários para tal, como GPU´s e ambientes em nuvem.

Outra resposta seria a desconexão entre profissionais de TI e cientistas de dados e engenheiros de Machine Learning. Profissinais de TI tendem a prover um ambiente estável e confiável, em contra partida profissinais que lidam com Machine Learning focam em iterações e experimentação.

Nesse post irei abordar os passos necessários para disponibilizar uma API de previsão de valores para imóveis treinada com Machine Learning. O post detalha o próximo passo após os passos iniciais do fluxo de um projeto de Machine Learning: entendimento do problema, aquisição e tratamento dos dados; criação de hipóteses; definição, avaliação e validação do modelo de machine learning; validação das hipóteses.

![Machine Learning Flow](/assets/img/deploy2.gif)

Nesse post iremos ver como preparar um ambiente em Windows para desenvolvimento de software, na passo seguinte irei abordar brevemente o modelo de Machine Learning criado e de forma mais abrangente a sequência para que esse modelo alimente uma API online.

Todo o conteúdo desenvolvido está sincronizado com o [repositório](https://github.com/mabittar/imovsp/tree/master) desse projeto.
Deixo aqui me reconhecimento e agradeciamento ao curso que provocou esse posto e outros projetos mais sobre o curso pode ser visto em: [https://sigmoidal.ai](https://sigmoidal.ai).

## 1. Preparação do ambiente

Em um ponto ou outro quando trabalhamos com desenvolvimento de software acabamos por nos deparar com ambientes virtuais. Essa é uma boa prática e altamente recomendável.

Desenvolver um aplicativo limita você a uma versão específica da linguagem e das bilbiotecas que utilizou para o desenvolvimento, instalados em seu sistema operacional. Tentar executar todos os aplicativos em uma única instalação torna provável que ocorram conflitos de versões entre coleções de bibliotecas. Também é possível que mudanças no sistema operacional quebrem outros recursos de desenvolviment que dependem dele.

Em um ambiente virtual, ou `virtualenv` são instalações leves e independentes, projetadas para serem configuradas com o mínimo de confusão e "simplesmente funcionar" sem exigir configuração extensiva ou conhecimento especializado.

O `virtualenv` evita a necessidade de instalar pacotes globalmente. Quando um virtualenv está ativo, em Python pro exemplo, o pip instala pacotes no ambiente, o que não afeta a instalação base do Python que foi realizada no sistema operacional de forma alguma.

### 1.1 O Plano

O nosso plano será preparar um ambiente windows de desenvolvimento, para criarmos um modelo de Machine Learning, a partir de um notebook criado no ambiente virtual de forma a receber novas consultas no formato json e retornar um valor de previsão de venda para um imóvel em São Paulo. 

A paritr do notebook iremos exportar o modelo de Machine Learning. Com o [Insomnia](https://insomnia.rest/) iremos certificarnos que a API está recebendo corretametne os dados e retornando  valor desejado. Com a confirmação do teste, iremos exportar a API para o ambiente web utilizando o [Heroku](https://heroku.com/) a fim de disponibilizarmos para consultas na online de forma independente do ambiente virtual que .

### 1.2 Instalação do Python

Após diversas tentativas com erros e acertos acabei esbarrando em um tutorial que me auxiliou muito durante esse processo:

Para alguns pode complicar um pouco, pois o tutorial está em inglês, porém a boa notícia que é um vídeo então fica fácil de acompanhar as etapas corretas de instalação:

O vídeo pode ser acessado diretamente neste [link](https://youtu.be/28eLP22SMTA), vou reproduzir aqui algumas anotações que fiz:

 - Durante a instalção não adicione o python ao PATH do windows. A princípio pode ser um incômodo toda vez que for utilziar o comando python para iniciar uma aplicação será necessário digitar o caminho completo, por exemplo: `c:\python\386\python` mas repare que dessa forma vc é capaz de escolher qual versão do python desejará utilizar e se você manter o padrão aos selecionar a pasta de instalação poderá ter mais de um versão em seu computador, sem comprometer o que já havia sido criado.

  + Com o python instalado, a partir da versão 3.3, já é possível criar prontamente um ambiente virtual, pois já possui com as bibliotecas necessárias. Para tanto no terminal do windows (tecla windows + r -> digite cmd.exe e tecle enter), acesse a pasta onde deseja criar o seu projeto. Aqui fica mais uma dica para que você crie uma pasta de projetos no seu hd, por exemplo: `c:\pyprojeto` para acessar a pasta digite `cd\pyprojeto`, uma vez dentro da pasta entre com o seguinte comando para criar o ambiente virtual `c:\pasta_de_instalação_python\python -m venv nome_projeto` substituindo pasta_de_instalação_python pelo caminho onde seu python foi instalado e no lugar de nome_projeto o nome do seu projeto, para mim ficou: `c:\python\386\python -m venv imovsp`. Aguarde um tempo, pois seu projeto estará em criação.

   - Após a criação do ambiente virtual será necessário acessá-lo, para acessá-lo será necessário mais um comando: `nome_projeto\Scripts\activate` , mais uma vez será necessário substituir name_projeto pelo nome que você deu ao seu projeto, no meu caso ficou assim: `imovsp\Scripts\activate` .

### 1.3 Visual Studio Code ou VSCode

Largamente utiizado pela industria o [VSCode](https://code.visualstudio.com/) é a IDE desenvolvida pela Microsoft com uma série de facilidades embutidas.

Todas as etapas que vimos no passo 1.1 podem se feitas diretamente no VSCode. Como há muitas referênciaspara configuração do ambiente na web e a própria documentação desenvolvida pela MS é ampla, vou passar apenas pelos pontos que tive dificuldade.

# Conclusão

Pode-se observar que mesmo com um dataset tratado, sem outliers ou valores ausentes, normalizado e com dados codificados não se trata de um problema trivial. 

Além disso, nossos dados de amostra se mostraram insuficientes, pois nosso modelo não consegue detectar corretamente muitos casos de `default` e, em vez disso, classifica incorretamente casos onde não ocorreriam. Característica de um dataset desbalanceado.

Da documentação oficial do [TensorFlow](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#applying_this_tutorial_to_your_problem) é possível extrair o seguinte trecho:

    "A classificação de dados desbalanceados é uma tarefa inerentemente difícil, pois há tão poucos exemplos para aprender. Devemos sempre começar com os dados primeiro e fazer o seu melhor para coletar o maior número possível de amostras e pensar bastante sobre quais recursos podem ser relevantes para que o modelo possa obter o máximo de sua classe minoritária. Em algum ponto, o modelo pode ter dificuldades para melhorar e produzir os resultados desejados, portanto, é importante ter em mente o contexto do seu problema e as compensações entre os diferentes tipos de erros."

Imagine que um modelo deste poderia elevar em muito as provisões utilizadas pelo banco para cobrir casos em que o cliente não cumpre com suas obrigações financeiras, elevando assim o custo de capital da instituição patrocinadora Nubank e consequentemente a taxa de juros a ser cobrada em cada empréstimo do cliente final.

Conforme foi possível obsevar durante o desenvolvimento dos modelos algumas variáveis possuem mais ou menos peso em determinado modelo, porém as métricas que influenciam os modelos aqui desenvolvidos, com os dados disponíveis são: `income` e `score_` de crédito. 

Grandes empresas de crédito implementam áreas robustas para desenvolver modelos e ajustá-los conforme crescimento da base de dados. Esse é um campo muito fértil e financeiramente viável de estudos relacionados a Data Science e Machine Learning.
