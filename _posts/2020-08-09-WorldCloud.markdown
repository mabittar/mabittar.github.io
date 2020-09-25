---
layout: post
title: WorldCloud
date: 2020-08-09 00:00:00 +0300
description: Utilizando Python e a biblioteca wordcloud para criar uma nuvem de palavras.. # Add post description (optional)
img: Pandas.jpeg # Add image post (optional)
tags: [Pandas, Cloud, Apresentação, Colab, Google] # add tag
---

A apresentação de dados é um dos grandes desafios para quem prepara apresentações. Normalmente vemos inúmeros ppt´s com aqueles gráficos complexos que demandam praticamente uma graduação para compreendê-los.

Imagine o caso que lhe fosse pedido para analisar as palavras que aparecem repetidamente nas descrições de imóveis do AirBnB, que trabalho seria fazer no excel e depois transformar em um gráfico. Muitas vezes as ferramentas disponíveis nos pacotes pré-instalados no computador são limitadas.
Mas no python fica muito mais fácil, veja um exemplo:

![](https://miro.medium.com/max/556/1*LXRvqD389NScGJtyDfNYog.png){: .center-image }

{% highlight python %}
import numpy as np
from PIL import Image

# endereço LOCAL da SUA imagem
sing_image = np.array(Image.open("/content/merlion-singapore.jpg"))
    
# gerar uma wordcloud
wordcloud3 = WordCloud(stopwords=stopwords2,
                        background_color="black",
                        width=1000, height=1000, max_words=500,
                        mask=sing_image, max_font_size=200,
                        min_font_size=.5, contour_width=3, contour_color='steelblue').generate(all_summary2)
    
# mostrar a imagem final
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(wordcloud3, interpolation='bilinear')
ax.set_axis_off()
    
plt.imshow(wordcloud3);
wordcloud.to_file("airbnb_summary_wordcloud.png")
{% endhighlight %}

Com poucas linhas de códigos é possível criar muito mais. Veja no notebook onde eu apresento os passos para preparar um nuvem de palavras ou tags utilizando Python e a biblioteca wordcloud.

Acesse o [notebook](https://colab.research.google.com/drive/1SSGPl-BWrrPENhPKhEHYW8TfLbOwBHWK?usp=sharing) para conferir os detalhes.


Veja outro exemplo do que podemos criar

![](https://miro.medium.com/max/425/1*3HFD6KgKLaExqs3VabgyJA.png){: .center-image }


**Data Science na Prática**

O material aqui desenvolvido é parte da provocação feita no curso de Data Science na Prática onde fui desafiado a tentar explicar os passos e ferramentas aplicadas durante a evolução do material.
Todo o material a ser desenvolvido no curso será centralizado no GitHub. 

[https://sigmoidal.ai](https://sigmoidal.ai)







