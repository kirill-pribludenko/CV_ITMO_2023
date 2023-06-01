В курсе требовалось развернуть S3 хранилище + DVC. Но к счастью(или сожалению) в нашем проекте данный функционал реализован с помощью **ClearL**

На данной картинке видно, что в нашем проекте хранятся разные версии в хранилище **ClearML**
 
 
<a href="/MLOps_course/DVC+S3/Example_of_datasets_1.png"><img src="/MLOps_course/DVC+S3/Example_of_datasets_1.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


На данной картинке видно, что есть два разных датасета для нашего проекта
 
 
<a href="/MLOps_course/DVC+S3/Example_of_datasets_2.png"><img src="/MLOps_course/DVC+S3/Example_of_datasets_2.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


На данной картинке видно, что при обучении модельки мы обращаемся к нашему датасету из хранилища **ClearML** и делаем локальную копию данных. 
   

<a href="/MLOps_course/DVC+S3/Example_of_datasets_3.png"><img src="/MLOps_course/DVC+S3/Example_of_datasets_3.png" style="width: 500px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


Так же в файле `/src/data/upload_data_to_clearml.py` можно найти каким образом загружались данные в хранилище **ClearML**