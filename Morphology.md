# Продвинутые задачи и модели NLP. Морфология
Данный курс посвящен основным задачам морфологии и способам их решения.

 Будут рассмотрены:
- Основные понятия 
- Готовые инструменты
- Способы решения задач
- Deep learning подходы

Старые [презентации](https://drive.google.com/drive/u/0/folders/0B07mCd0ZUKkjb1lCMmxrQXlYNGc) Ильи Гусева.


<div style="page-break-after: always;"></div> 


## Структура курса
<!-- TOC -->
- Введение
    - [Основные понятия](./basics/Basics.md)
    - Морфологические парсеры
- [Part-of-speech tagging](http://nlpprogress.com/english/part-of-speech_tagging.html)
- Исправление ошибок
    - Исправление опечаток
    - [Grammatical error correction](http://nlpprogress.com/english/grammatical_error_correction.html)
- Морфологический синтез
    - [Lexical normalization](http://nlpprogress.com/english/lexical_normalization.html)
<!-- /TOC -->


<div style="page-break-after: always;"></div> 


## Основные понятия
**Морфология** - раздел грамматики, основными объектами которого являются  **слова** естественных языков, их значимые части и морфологические признаки.

<img src="pics/pyramid.PNG" height="300" style="float:right">

#### Морфология изучает:
- Словоизменение
- Словообразовние 
- Части речи
- Грамматические значения





<div style="page-break-after: always;"></div> 


## Задачи компьютерной морфологии
- **Спеллинг** - проверка слова по словарю
- **Морфологический анализ** - определение грамматического значения и леммы у словоформы
- **Морфологический синтез** - построение словоформы по лемме и грамматическому значению
- **Исправление опечаток** - поиск наиболее близкого слова в слоаваре