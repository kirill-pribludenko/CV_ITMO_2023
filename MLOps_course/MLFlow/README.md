В курсе требовалось развернуть MLFlow. Но к счастью(или сожалению) в нашем проекте данный функционал реализован с помощью **ClearL**

На данной картинке видно, сколько экспериментов было проведено в данном проекте **ClearML**
 
 
<a href="MLOps_course\MLFlow\Example_of_exp_1.png"><img src="MLOps_course\MLFlow\Example_of_exp_1.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


На данной картинке видно, что имея один и тот же файл для обучения модели, можно менять через веб-морду **ClearML** параметры обучения
 
 
<a href="MLOps_course\MLFlow\Example_of_exp_2.png"><img src="MLOps_course\MLFlow\Example_of_exp_2.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


На данной картинке видно, что после каждого эксперимента обучения у нас сохраняется модель, которую позже можно скачать.


<a href="MLOps_course\MLFlow\Example_of_exp_3.png"><img src="MLOps_course\MLFlow\Example_of_exp_3.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


На данной картинке видно, что для каждого эксперимента "рисуются" графики необходимых параметров.


<a href="MLOps_course\MLFlow\Example_of_exp_4.png"><img src="MLOps_course\MLFlow\Example_of_exp_4.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


Еще отмечу тот факт, что **ClearML** супер удобная штука. Можно подключать множество воркеров(более сильные машинки) к проекту, и вести обучение на них, при этом автоматически создается очередь обучения. К примеру, весь проект "писался" на ноутбуках, а обучение и эксперементы проводилось ночью на домашнем-мощном компе одного из участников команды.