from html import escape

from lxml import etree, html


def add_td_tag(contents, row_span, col_span):
    return f'<td rowspan="{row_span}" colspan="{col_span}">{contents}</td>'


def add_table_tag(contents):
    return f"<table border='1' style='border-collapse: collapse'>{contents}</table>"


def add_tr_tag(contents):
    return f"<tr>{contents}</tr>"


def add_p_tag(contents):
    return f"<p>{contents}</p>"


def add_html_tag(text):
    return f"<html><body>{text}</body></html>"


def export_html(inputs, out_path: str):
    html_string = ""
    elements = []
    for table in inputs.tables:
        pre_row = 1
        rows = []
        row = []
        for cell in table.cells:
            if cell.row != pre_row:
                rows.append(add_tr_tag("".join(row)))
                row = []

            row_span = cell.row_span
            col_span = cell.col_span
            contents = cell.contents

            if contents is None:
                contents = ""

            contents = escape(contents)

            row.append(add_td_tag(contents, row_span, col_span))
            pre_row = cell.row

        table_html = add_table_tag("".join(rows))
        elements.append(
            {
                "box": table.box,
                "html": table_html,
            }
        )

    for paraghraph in inputs.paragraphs:
        contents = escape(paraghraph.contents)

        elements.append(
            {
                "box": paraghraph.box,
                "html": add_p_tag(paraghraph.contents),
            }
        )

    elements = sorted(elements, key=lambda x: x["box"][1])
    html_string = "".join([element["html"] for element in elements])
    html_string = add_html_tag(html_string)

    parsed_html = html.fromstring(html_string)
    formatted_html = etree.tostring(
        parsed_html, pretty_print=True, encoding="unicode"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted_html)