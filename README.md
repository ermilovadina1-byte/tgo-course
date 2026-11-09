# Технологии глубокого обучения: лабораторные

[![tests](https://github.com/ermilovadina1-byte/tgo-course/actions/workflows/tests.yml/badge.svg)](https://github.com/ermilovadina1-byte/tgo-course/actions/workflows/tests.yml)

Дина Ермилова, МТУСИ, группа МТИ2531. Заочная программа `part-time`.

Репозиторий подготовлен на основе [шаблона преподавателя](https://github.com/pmenshih/tgo-course). [Условия заочного трека](part-time/README.md), [исходный README курса](COURSE_README.md).

| Работа | Результат |
|---|---|
| [pt01: счетчики и BPE](pt01/README.md) | Реализована, локальные проверки пройдены |
| pt02: трансформер | Не начата, сохранена заглушка из шаблона |

## Локальный запуск

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r pt01/requirements-dev.txt
.venv/bin/python pt01/run.py
.venv/bin/python -m pytest tests/test_pt01.py pt01/tests -q
```

Для pt01 вычисления используют только стандартную библиотеку Python 3.11+. Общий `requirements.txt` сохранен из курса для CI и следующих работ.

`python pt01/run.py --fast` делает 200 слияний BPE и перезаписывает артефакт. Перед коммитом результатов нужен полный запуск без `--fast`: он сохраняет 1 500 слияний. Корпус скачивается автоматически, в Git не входит. Веса и виртуальные окружения тоже исключены. Унаследованный `demos/02/corpus.txt` удален из этой копии шаблона; pt01 скачивает собственный корпус при запуске.

## Автоматические проверки

Workflow `.github/workflows/tests.yml` запускается при push или вручную через Actions. Он берет тесты и настройки pytest из актуального репозитория курса, выполняет `run.py --fast` и официальные тесты. Дополнительно для pt01 выполняются собственные тесты. Заглушка pt02 пропускается.

Временные артефакты быстрого запуска в CI остаются на runner и не заменяют полные результаты в репозитории. Состояние запусков доступно в [GitHub Actions](https://github.com/ermilovadina1-byte/tgo-course/actions/workflows/tests.yml).
