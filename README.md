# Технологии глубокого обучения: лабораторные

[![tests](https://github.com/ermilovadina1-byte/tgo-course/actions/workflows/tests.yml/badge.svg)](https://github.com/ermilovadina1-byte/tgo-course/actions/workflows/tests.yml)

Дина Ермилова, МТУСИ, группа МТИ2531. Заочная программа `part-time`.

Репозиторий подготовлен на основе [шаблона преподавателя](https://github.com/pmenshih/tgo-course). [Условия заочного трека](part-time/README.md), [исходный README курса](COURSE_README.md).

| Работа | Содержание |
|---|---|
| [pt01: счетчики и BPE](pt01/README.md) | Биграммная модель, перплексия, генерация и свой BPE |
| [pt02: трансформер](pt02/README.md) | Обучение декодера, режимы генерации и сравнение с биграммой |

## Локальный запуск

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r pt01/requirements-dev.txt
.venv/bin/python pt01/run.py
.venv/bin/python -m pytest tests/test_pt01.py pt01/tests -q
```

Нужен Python 3.11 или новее; в примере использован Python 3.12. Для pt01 вычисления используют только стандартную библиотеку. Общий `requirements.txt` сохранен из курса для CI.

Для pt02 используются правила BPE из первой работы:

```sh
.venv/bin/python -m pip install -r pt02/requirements.txt
.venv/bin/python pt02/run.py
.venv/bin/python -m pytest tests/test_pt02.py pt02/tests -q
```

Полный запуск pt02 выполняет 2000 шагов обучения, сохраняет кривую ошибки и образцы, сравнивает четыре режима генерации и оценивает трансформер вместе с биграммой на 2000 словах отложенного текста. Веса сохраняются локально в `pt02/model.pt` и исключены из Git. Они позволяют повторить отдельные эксперименты без нового обучения:

```sh
.venv/bin/python pt02/run.py --generate-only
.venv/bin/python pt02/run.py --evaluate-only
```

Флаг `--fast` сокращает pt01 до 200 слияний BPE, а pt02 до 200 шагов обучения и оценки на 300 словах. Он перезаписывает результаты соответствующей работы; для сохранения полного прогона pt02 можно задать отдельную папку через `--artifacts` и отдельный файл весов через `--checkpoint`. Корпус скачивается автоматически и не входит в Git. Папки `steps/` содержат небольшие учебные примеры.

## Автоматические проверки

Workflow `.github/workflows/tests.yml` запускается при push или вручную через Actions. Он берет тесты и настройки pytest из актуального репозитория курса, выполняет `run.py --fast`, официальные и собственные тесты. Изменение pt01 также запускает проверку pt02, поскольку вторая работа использует код и BPE из первой.

Временные артефакты быстрого запуска в CI остаются на runner и не заменяют полные результаты в репозитории. Состояние запусков доступно в [GitHub Actions](https://github.com/ermilovadina1-byte/tgo-course/actions/workflows/tests.yml).
