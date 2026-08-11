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


def test_reconstructs_captioned_bitfield_with_short_edge_cells():
    lines = [
        ([("Table 26. WHO_AM_I register default values", 80, 220)], 2, 10),
        (
            [
                ("X_L7", 10, 25),
                ("X_L6", 60, 75),
                ("X_L5", 110, 125),
                ("X_L4", 160, 175),
                ("X_L3", 210, 225),
                ("X_L2", 260, 275),
                ("0", 325, 330),
                ("0", 395, 400),
            ],
            18,
            26,
        ),
    ]

    html = reconstruct_sparse_table_html(lines, [0, 0, 410, 30])

    soup = BeautifulSoup(html, "html.parser")
    assert soup.caption.get_text() == "Table 26. WHO_AM_I register default values"
    assert table_cells(html) == ["X_L7", "X_L6", "X_L5", "X_L4", "X_L3", "X_L2", "0", "0"]


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


def test_reconstructs_spaced_uppercase_field_names():
    lines = [
        ([("FF_DUR [4:0]", 10, 55)], 10, 18),
        ([("Free-fall duration", 70, 150)], 7, 15),
        ([("FF_THS [2:0]", 10, 55), ("Free-fall threshold", 70, 155)], 30, 38),
    ]

    html = reconstruct_sparse_table_html(lines, [0, 0, 180, 45])

    assert table_cells(html) == [
        "FF_DUR[4:0]",
        "Free-fall duration",
        "FF_THS[2:0]",
        "Free-fall threshold",
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


def test_rejects_regular_multicolumn_table():
    lines = [
        (
            [
                ("Model", 10, 45),
                ("Temperature Range", 70, 140),
                ("Package Description", 170, 250),
                ("Quantity", 280, 320),
            ],
            10,
            18,
        ),
        (
            [
                ("ADXL380", 10, 45),
                ("-40 to 125 C", 70, 135),
                ("14-Terminal LGA", 170, 245),
                ("5000", 285, 310),
            ],
            30,
            38,
        ),
    ]

    assert reconstruct_sparse_table_html(lines, [0, 0, 340, 45]) is None


def test_rejects_prose():
    lines = [
        ([("This is a sentence, not a sparse table.", 10, 170)], 10, 18),
        ([("It should remain ordinary text.", 10, 150)], 20, 28),
    ]

    assert reconstruct_sparse_table_html(lines, [0, 0, 180, 30]) is None
