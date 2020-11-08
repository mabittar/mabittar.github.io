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

Um artigo muito interessante sobre a dificuldade desta etapa pode ser visto no post do [KDnuggtes](https://www.kdnuggets.com/2019/10/machine-learning-deployment-hard.html).

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

   - Após a criação do ambiente virtual será necessário ativá-lo, para isso entre com o comando: `nome_projeto\Scripts\activate` , mais uma vez será necessário substituir name_projeto pelo nome que você deu ao seu projeto, no meu caso ficou assim: `imovsp\Scripts\activate`. Você poderá obserar na linha de comando que o nome do projeto estará entre parênteses `(imvosp) c:\pyprojeto\imovsp` isso significa que o ambiente virtual está ativado e as bibliotecas que futuramente forem instaladas utilizando o comando pip, por exemplo: `pip install pandas numpy` ficarão restritas a esse ambiente.

   - Aproveito para utilizar o gancho aqui, que ao final do desenvolvimento iremos utilizar o comando `pip freeze > requirements.txt` a fim de gerar um lista de todas as bibliotecas que utilizamos durante o desenvolvimento.

   - O python irá ser acionado com o comando direto `python`, após o enter você verá o prompt inciaindo com  >>. Para desativar o ambiente virtual basta entrar com o comando `deactivate`.

### 1.3 Visual Studio Code ou VSCode

Largamente utiizado pela industria o [VSCode](https://code.visualstudio.com/) é a IDE desenvolvida pela Microsoft com uma série de facilidades embutidas.

Todas as etapas que vimos no passo 1.1 podem se feitas diretamente no VSCode. Como há muitas referênciaspara configuração do ambiente na web e a própria documentação desenvolvida pela MS é ampla, vou passar apenas pelos pontos que tive dificuldade.

Vale ressaltar que a principal facilidade do VSCode é trabalhar com diversas extensões, criando uma infinidade de facilidades. Se você ainda não o utilizou após as atualizações em 2019 vale conhecê-lo ou mesmo se voc~e utiliza outras IDE´s vai ficar impressinado com as facilidades de importação de todas as configurção para a nova IDE.

Em relação as dificuldades encontradas posso destacar principalmente a minha falta de experiência para lidar com o software me si, apesar da minha vontade em aprender programação e disciplinas ligadas a inteligência artifical, estava muito acustumado com o ambiente do [Google Colab](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb). Sair da zona de conforto nos possibilita novos aprendizados e favorece o nosso desenvolvimento.

Novamente apoiei em um dos vídeos do Corey Schafer [Setting up a Python Development Environment](https://youtu.be/-nh9rCzPJ20), o vídeo possui mais de uma 1hora de gravação, bem extenso passando por diversas possibilidades com muito detalhes e dicas, vale muito a pena para evitar algumas dores de cabeça.


### Notebook

[notebook](https://github.com/mabittar/imovsp/blob/master/model.ipynb)


### Imsomnia

[documentação oficial](https://hcode.com.br/blog/usando-insomnia-para-testar-as-requisicoes-de-nossas-apis) para utilização do Imsonia

Criar get
criar post

![Test API](/assets/imgs/deploy-test-post.JPG)


### Heroku


Os passos para enviar a API testada para o Heroku são:


Criar um arquivo "Procfile"
Atualizar o Procfile com web: gunicorn app:app
pip3 freeze > requirements.txt
git init
heroku login
heroku create nomedasuaapp
git add .
git commit -m "Text do Commit"
heroku git:remote -a nomedasuaapp
git push heroku master

Durante o upload dos arquivos utilizando o git push tive diversas dificuldades, pois algumas bilbiotecas que eu havia utilizado para desenvolvimento não estavam disponíveis no Heroku, mas com um olhar mais atento pude perceber que tais bibliotecas foram instaladas pelo VSCode e não iriam interferir com o funcionamento do modelo desenvolvido. Para tanto alterei no arquivo `requirements.txt` as versões das bibliotecas que eu utilizei com as que estavam disponíveis no momento em que eu fiz o deploy no Heroku. Na mesma linha de comando que você acompanha o envio dos arquivos para a web será possível observar quais as versões das bibliotecas que estão disponíveis. Tal erro vai aparecer após o envio dos arquivos, quando o Heroky estiver montando a aplicação.

Apenas como exemplo, quando eu criei o documento requirements.txt estava com a biblioteca pylint na versão 3.3, porém o Heroku possuia apenas a versão 2.6.0, bastou alterar manualmente no arquivo e realizar os passos que esse erro foi superado.

[link](https://imovsp.herokyapp.com/)


# Conclusão

Apesar de tudo a cadeia de Machine Learning ainda está em seus estágios iniciais. Na verdade, os componentes de software e hardware estão em constante evolução para atender às demandas atuais do ML.

Docker / Kubernetes e arquitetura de microsserviços podem ser empregados para resolver os desafios de heterogeneidade e infraestrutura. As ferramentas existentes podem resolver muito alguns problemas individualmente. 

Implantar Machine Learning em empresas é e continuará sendo difícil, e isso é apenas uma realidade com a qual as organizações precisarão lidar. Felizmente, algumas novas arquiteturas e produtos estão ajudando os cientistas de dados. Além disso, à medida que mais empresas estão escalando as operações de ciência de dados, elas também implementam ferramentas que tornam a implantação do modelo mais fácil.

Em empresas internacionais acredito que reunir todas essas ferramentas para operacionalizar o ML é o maior desafio hoje, porém em empresas nacionais continua sendo a questão de aquisição e tratamento de dados, portanto ainda estamos engatinhando no processo.
