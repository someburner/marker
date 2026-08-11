from bs4 import BeautifulSoup

from marker.processors.table_recon import reconstruct_sparse_table_html


def table_cells(html):
    return [
        cell.get_text() for cell in BeautifulSoup(html, "html.parser").find_all("td")
    ]


def test_reconstructs_wrapped_register_bitfield():
    lines = [
        (
            [("DRDY_", 10, 30), ("INT2_ON_", 70, 100), ("RESERVED", 130, 160)],
            10,
            18,
        ),
        ([("PULSED", 9, 31), ("INT1", 77, 93), ("0", 145, 150)], 19, 27),
    ]

    html = reconstruct_sparse_table_html(lines, [0, 0, 180, 30])

    assert table_cells(html) == ["DRDY_PULSED", "INT2_ON_INT1", "RESERVED0"]


def test_reconstructs_field_description_rows():
    lines = [
        ([("FIELD_A", 10, 45)], 10, 18),
        ([("First field description", 70, 160)], 7, 15),
        ([("continued", 70, 110)], 19, 27),
        ([("FIELD_B", 10, 45), ("Second description", 70, 150)], 35, 43),
    ]

    html = reconstruct_sparse_table_html(lines, [0, 0, 180, 50])

    assert table_cells(html) == [
        "FIELD_A",
        "First field description continued",
        "FIELD_B",
        "Second description",
    ]


def test_reconstructs_touching_prose_and_unit_spans():
    lines = [
        (
            [
                ("FIELD_A", 10, 45),
                ("FDS bit in", 70, 105),
                ("CTRL6 (25h)", 105, 150),
                ("must be set", 150, 190),
            ],
            10,
            18,
        ),
        (
            [
                ("FIELD_B", 10, 45),
                ("977 µ", 70, 95),
                ("g", 95, 100),
                ("/LSB", 100, 120),
            ],
            30,
            38,
        ),
    ]

    html = reconstruct_sparse_table_html(lines, [0, 0, 200, 45])

    assert table_cells(html) == [
        "FIELD_A",
        "FDS bit in CTRL6 (25h) must be set",
        "FIELD_B",
        "977 µg/LSB",
    ]


def test_rejects_prose():
    lines = [
        ([("This is a sentence, not a sparse table.", 10, 170)], 10, 18),
        ([("It should remain ordinary text.", 10, 150)], 20, 28),
    ]

    assert reconstruct_sparse_table_html(lines, [0, 0, 180, 30]) is None
