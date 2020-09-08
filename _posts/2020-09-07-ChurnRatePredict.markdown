---
layout: post
title: Churn Rate Predict
date: 2020-09-07 00:00:00 +0300
description: Analisando Churn Rate com diferentes métodos de ML -  # Add post description (optional)
img: churn-rate.jpg # Add image post (optional)
tags: [Machine Learning, Otmização, XGboost, SGD, SVM, RandomTree, Classifiers, SHAP, Grid Search, Bayes Search] # add tag
---

## Churn Rate
Se você vende o mesmo produto que seu vizinho vende, e ainda, por um preço melhor, por que o cliente não fica com o seu produto?

Entender por que seus clientes abandonam o seu produto ou serviço é vital para conquistar um crescimento sustentável. Como o Churn tem um efeito negativo na receita de uma empresa, entender o que é esse indicador e como trabalhar para mitigar essa métrica é algo crítico para o sucesso de muitos negócios.

O que é Churn Rate? Em uma tradução simples Churn Rate é a taxa de cancelamento, ou de abandono, registrada em sua base de clientes. por exemplo, para setores de serviço significa o cancelamento do serviço.

Embora tenha como principal função medir o percentual de clientes que abandonam um serviço, também serve para evidenciar o impacto negativo desses cancelamentos no caixa. Para alguns setores, esta é uma métrica básica para avaliar o sucesso do negócio, já que apresenta impacto direto no faturamento. Este é o caso dos serviços de assinatura.

Esse estudo é uma provocação feita no curso Data Science na Prática onde fui desafiado a tentar explicar os passos e ferramentas aplicadas durante a evolução do material.
Todo o material a ser desenvolvido no curso será centralizado no meu [portfolio de projetos](https://github.com/mabittar/Portfolio). 
Mais sobre o curso pode ser visto em: [https://sigmoidal.ai](https://sigmoidal.ai).

## O estudo
No [notebook](https://colab.research.google.com/drive/1JFs_T1AJTsg7KqlHlQnJRW59xx3rrzMD?usp=sharing) foi elaborado um passo a passo detalhado para que seja possível replicar a analise dos dados disponível. Foram utilizados diferentes modelos de Apredizado de Máquinas. Iniciei os estudos fazendo primeiramente o  estudo e verificação de equilíbrio das variáveis e dados do dataset.

![](assets/img/churn-correlation.jpg)


## Diferentes modelos de Machine Learning

Primeiramente foi utilizado um modelo de Árvore de Decisões, pois não seria necessário realizar outras intervenções (padronizar e normalizar o dataset) e assim seria possível para obter um paramêtro para verificar em relação aos demais modelos.

{% highlight python %}

#1. Escolha do modelo
from sklearn.ensemble import RandomForestClassifier

#2. Instanciar o modelo e optimizar hiperparâmetros
model_rf = RandomForestClassifier(max_depth=4, random_state=42)

#Separando os dados em features matrix e target vector

#3.1 Dividir o dataset entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#4. Fit do modelo
model_rf.fit(X_train, y_train)

#5. Previsões
y_pred_rf = model_rf.predict(X_test)
{% endhighlight %}

com esse modelo foi possível obter:

![](assets/img/churn-report-DTC.JPG)

## Demais modelos

Com os resultados obtidos no modelo inicial, os próximos passos seriam desenvolver novos modelos e compará-los com o modelo base a fim de comparar e verificar o desempenho.

Como explicado no [post](_posts/2020-08-31-EnsembleMethod.markdown) utilizei o método Ensemble para desenvolver paralelamente diversos modelos. A partir foi utilizado a métrica de votação `hard` para obtermos o resultado final desta etapa:

Deixarei no post o detalhamento utilizado da método:

{% highlight python %}
# 1. importando bibliotecas necessárias
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# 2. Instanciar os modelos e definir hyperparametros

model_xgbc = XGBClassifier()
model_sgd = SGDClassifier()
model_svc = SVC()
model_lr = LogisticRegression()
model_dt = DecisionTreeClassifier()
voting_clf = VotingClassifier(estimators =[('xgbc', model_xgbc), ('sgd', model_sgd), ('svc', model_svc),('dt', model_dt), ('lr', model_lr)], voting='hard')

# 3. Separar os dados
#os dados já foram separados anteiormente

# 3.1 Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# 4. Fit do modelo
for model in (model_xgbc, model_sgd, model_svc, model_dt,model_lr, voting_clf):
  model.fit(X_train_scaled, y_train)

# 5. Fazer previsões em cima dos dados
model = []
accuracy = []

for clf in (model_xgbc, model_sgd, model_svc, model_dt, model_lr, voting_clf):
  y_pred = clf.predict(X_test_scaled)
  model.append(clf.__class__.__name__)
  accuracy.append(accuracy_score(y_test,y_pred))

# 6. Verifcando Acurácia dos modelos
col = ['Acuracia']
ac = pd.DataFrame(data=accuracy, index=model, columns=col)
ac

{% endhighlight %}


A partir do modelo acima os resultados de acurácia obtidos foram:

![Acurácia](assets/img/chrun-ensemble.JPG)


Após a elaboração dos modelos paralelos foi utilizado o processo de `Validação Cruzada` para otimizar a amostra de dados e tentar obter uma melhora nos resultados

{% highlight python %}
# Aplicando Validação Cruzada com K-Fold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = voting_clf, X = X_train_scaled, y = y_train, cv = 10, verbose=3)
print("Acurácia: {:.2f} %".format(accuracies.mean()*100))

{% endhighlight %}


Com a acurácia do modelo em torno de 80% os próximos passos foram otimizar os hiperparâmetros.

## Otimização do Modelo

Nos próximos passos utilizei o modelo `XGBoost Classifier` e executei algumas rotinas de otimização dos hiperparâmetros.

### Grid Search

O método Grid Search é uma abordagem de força bruta que testa todas as combinações de hiperparâmetros para encontrar o melhor modelo. Seu tempo de execução explode com o número de valores (e combinações dos mesmos) para testar.

Ao executar os testes durante a elaboração deste notebook exagerei nas combinações possíveis e levei quase 2 horas para que chegasse no valor final.

{% highlight python %}

# 1. importar bibliotecas necessárias
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# 2.Instanciar o modelo
xgb = XGBClassifier()

# 3. definir intervalos para otimização
# 3.1 Cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)
#3.2 Intervalos para otimização
param_grid = {
    'xgb__n_components': [0, 1, 5],
    'xgb__n_estimators': range(0,200,50),
    'xgb__learning_rate': [0.001, 0.01, 0.05],
    'xgb__max_depth': [1, 2, 5],
    'xgb__gamma': [0.01, 0.1, 1]
}

# 3.3 gerar o modelo
grid = GridSearchCV(xgb, param_grid, n_jobs=-1, scoring='recall', cv=kfold, verbose=3)

# 4. Fit do modelo
grid_result = grid.fit(X_train_scaled, y_train)


# 5. ver resultados
print("Melhor acurácia: {} para {}".format(grid_result.best_score_, grid_result.best_params_))
{% endhighlight %}

### Bayes Search

Uma abordagem bayesiana diminui a probabilidade de que os valores escolhidos para o segundo modelo sejam parte da solução ótima. Agora ele usa as probabilidades atualizadas para selecionar um novo conjunto de valores para cada hiperparâmetro, ver se aumentou ou diminuiu a qualidade do modelo e atualizar as probabilidades. Em outras palavras, é mais provável que o algoritmo escolha valores para a próxima rodada que estão relacionados a um desempenho de modelo superior do que suas alternativas menos eficazes.

{% highlight python %}

# 1. importar as bibliotecas necessárias
import skopt
from skopt import BayesSearchCV

# 2. Definir intervalos de otimziação
bayes = BayesSearchCV(
    estimator = XGBClassifier(
        n_jobs = 1,
        eval_metric = 'auc',
        silent=1,
        tree_method='auto'
    ),
    # 2.1 Definindo intervalos de otimização
    search_spaces = {
        'learning_rate': (0.001, 1.0, 'log-uniform'),
        'min_child_weight': (range(1,5,1)),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    # 2. definindo método de avaliação
    scoring = 'recall',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = 6,   
    verbose = 0,
    refit = True,
    random_state = 42
)

# callback handler
def status_print_bayes(optim_result):
    bayes_resultado = bayes.best_score_
    print("Melhor resultado: %s", np.round(bayes_resultado,4))
    if bayes_resultado >= 0.98:
        print('Suficiente!')
        return True

# 4. Fit no modelo
otmizador = bayes.fit(X_train_scaled, y_train, callback=status_print_bayes)

{% endhighlight %}

A partir da otimização Bayes os parâmetros que resultaram na melhor classificação foram:
('colsample_bylevel', 0.4160029192647807), 
('colsample_bytree', 0.7304484857455519), 
('gamma', 0.13031389926541354), 
('learning_rate', 0.00885928719200224), 
('max_delta_step', 13), 
('max_depth', 21), 
('min_child_weight', 2), 
('n_estimators', 87), 
('reg_alpha', 5.497557739289786e-07), 
('reg_lambda', 648), 
('scale_pos_weight', 275), 
('subsample', 0.13556548021189216)

Os resultados obtidos nos dois passos anteriores serviram para definir o modelo final

{% highlight python %}
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.metrics import roc_auc_score, roc_curve

# modelo final
modelo_final = XGBClassifier(
    learning_rate=0.00885928719200224 , 
    n_estimators=87,
    max_delta_step = 13, 
    max_depth=1, 
    min_child_weight=2, 
    gamma=0.13031389926541354,
    colsample_bylevel = 0.4160029192647807,
    colsample_bytree = 0.7304484857455519,
    reg_lambda= 648

    )

modelo_final.fit(X_train_rus, y_train_rus)

# fazer a previsão
X_test = scaler.transform(X_test)
y_pred = modelo_final.predict(X_test)

# Classification Report
print(classification_report(y_test, y_pred))

# imprimir a área sob a curva
print("AUC: {:.4f}\n".format(roc_auc_score(y_test, y_pred)))

# plotar matriz de confusão
plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()
{% endhighlight %}


Com os dados desse modelo foi possível obter os seguintes resultados:
![](assets/img/Churn-report-final.JPG)


## Verificando a Influência de cada Variável no modelo
SHAP (SHapley Additive exPlanations) é uma abordagem teórica de jogos para explicar a saída de qualquer modelo de Machine Learning. Ele conecta a alocação de peso ideal com explicações locais usando os valores clássicos de Shapley da teoria dos jogos e suas extensões relacionadas.

![](assets/img/churn-shap.jpg)

Portanto as variáveis que mais influenciaram no nosso modelo são:
  - 33. Tipo de Contrato: Month to Month - contrato mensal
  - 15. Online Security: No - o cliente não possui serviço adicional de segurança online
  - 24. Tech Suport: No - o cliente não possui serviço adicional de suporte técnico
  - 4. Ternure: conforme já observado novos clientes estão mais propensos a cancelar o serviço

No [notebook](https://colab.research.google.com/drive/1JFs_T1AJTsg7KqlHlQnJRW59xx3rrzMD?usp=sharing) elaborado demonstros os passos de utilização dessa ferramenta incrível.

# Conclusão

Compreendendo melhor o churn rate, gestores e equipes estarão mais bem preparados para reduzir os índices de cancelamentos e, assim, ampliar a sua base de clientes.

No modelo matemático criado a partir da base de dados fornecida foi possível observar que clientes com contratos mensais, sem serviços adicionais como segurança online e suporte técnico são mais suscetíveis ao cancelamento do serviço. Com os insights obtidos a empresa pode direcionar seus esforços e custos em pontos críticos e assim modificar o `Churn Rate`.





