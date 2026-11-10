from lab02.run import select_paragraphs, span_positions, ENGLISH


def test_paragraph_boundaries_and_original_case():
    short='Ёж ' * 39
    first='Ёж ' * 40
    last='Жук ' * 120
    long='Жук ' * 121
    assert select_paragraphs('\n'.join([short,first,last,long]),500)==[
        {'id':0,'text':first.strip()},{'id':1,'text':last.strip()}]
    assert len(select_paragraphs('\n'.join([first,last]),1))==1


def test_overlapping_offsets_include_every_word_piece():
    offsets=[(0,2),(2,4),(4,8),(8,9)]
    assert span_positions(offsets,3,8)==[1,2]
    assert span_positions(offsets,8,9)==[3]


def test_english_matches_assignment():
    from pathlib import Path
    expected=Path(__file__).resolve().parents[2]/'tests/data/lab02/en_paragraph.txt'
    assert ENGLISH==expected.read_text().strip()
