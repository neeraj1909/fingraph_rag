"""Deterministic, repository-owned PDF bytes for the Docling integration test."""

from __future__ import annotations


def synthetic_acme_10q_pdf() -> bytes:
    """Build a small text-and-rule PDF without an external document corpus."""

    newline = b"\n"
    commands = [
        "BT /F1 12 Tf 72 740 Td (Synthetic Acme Corporation Form 10-Q) Tj ET",
        "BT /F1 11 Tf 72 710 Td (Item 2. Management Discussion and Analysis) Tj ET",
        "BT /F1 10 Tf 72 680 Td (Supplier concentration affected quarterly operations.) Tj ET",
        "0.5 w",
        "72 620 m 420 620 l S",
        "72 580 m 420 580 l S",
        "72 540 m 420 540 l S",
        "72 500 m 420 500 l S",
        "72 620 m 72 500 l S",
        "220 620 m 220 500 l S",
        "320 620 m 320 500 l S",
        "420 620 m 420 500 l S",
        "BT /F1 9 Tf 84 595 Td (Metric) Tj ET",
        "BT /F1 9 Tf 232 595 Td (Q3 2024) Tj ET",
        "BT /F1 9 Tf 332 595 Td (Q2 2024) Tj ET",
        "BT /F1 9 Tf 84 555 Td (Operating margin) Tj ET",
        "BT /F1 9 Tf 232 555 Td (18.4%[1]) Tj ET",
        "BT /F1 9 Tf 332 555 Td (17.9%) Tj ET",
        "BT /F1 9 Tf 84 515 Td (Supplier concentration) Tj ET",
        "BT /F1 9 Tf 232 515 Td (42%) Tj ET",
        "BT /F1 9 Tf 332 515 Td (39%) Tj ET",
        "BT /F1 9 Tf 72 460 Td ([1] Margin excludes restructuring costs.) Tj ET",
    ]
    body = newline.join(command.encode() for command in commands)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(body)).encode()
        + b" >>"
        + newline
        + b"stream"
        + newline
        + body
        + newline
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4" + newline)
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj".encode() + newline + obj + newline + b"endobj" + newline)
    xref_offset = len(pdf)
    pdf.extend(f"xref{newline.decode()}0 {len(objects) + 1}{newline.decode()}".encode())
    pdf.extend(b"0000000000 65535 f " + newline)
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n ".encode() + newline)
    pdf.extend(
        f"trailer{newline.decode()}<< /Size {len(objects) + 1} /Root 1 0 R >>{newline.decode()}"
        f"startxref{newline.decode()}{xref_offset}{newline.decode()}%%EOF{newline.decode()}".encode()
    )
    return bytes(pdf)
