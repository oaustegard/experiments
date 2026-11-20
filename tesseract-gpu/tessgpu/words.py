"""Word boxes from Tesseract's character positions.

    from tessgpu import words
    ids, xs = beam.decode_with_x(probs, model)
    ws = words.line_words(ids, xs, model.unicharset, line_box, crop_box, model.input_height)
    ws                                  # [{"text": "Net", "box": [x0, y0, x1, y1]}, ...]

A recognised line is split into words at the space unichar " ". Each character's left edge is
its first network timestep (xcoords from beam.decode_with_x). One timestep covers 3 pixels of
the line scaled to the model height, so with crop_box the padded crop the recogniser saw
(server/pipeline.py pads each line box by PAD px before cropping) and the model height H:

    page_x(t) = crop_box.x0 + t * 3 * crop_h / H

A word starts at the left edge of its first character and ends where the space that follows
it starts. The last word of a line ends at the line box's right edge, since there is no
following space to mark where its glyphs stop.
"""

TIMESTEP_PX = 3   # pixels of the line, at model height, covered by one network timestep
SPACE = " "


def char_left(t, crop_box, model_h):
    """Page x of network timestep t for a crop recognised at model height model_h."""
    crop_h = crop_box[3] - crop_box[1]
    return crop_box[0] + t * TIMESTEP_PX * crop_h / model_h


def line_words(ids, xcoords, unicharset, line_box, crop_box, model_h):
    """Split one line's unichar ids into words with page-pixel boxes, left to right.

    ids, xcoords  from beam.decode_with_x (len(xcoords) == len(ids))
    unicharset    model.unicharset, index -> unichar string
    line_box      [x0, y0, x1, y1] of the line in page pixels (unpadded)
    crop_box      [x0, y0, x1, y1] of the padded crop the recogniser saw
    model_h       model.input_height (the network's line height, 36 for these models)
    """
    if len(ids) != len(xcoords):
        raise ValueError("ids and xcoords differ in length: %d vs %d" % (len(ids), len(xcoords)))
    y0, y1 = line_box[1], line_box[3]
    words = []
    cur = None                      # [text, left]
    for uid, t in zip(ids, xcoords):
        ch = unicharset[uid] if 0 <= uid < len(unicharset) else "<%d>" % uid
        x = char_left(t, crop_box, model_h)
        if ch == SPACE:
            if cur is not None:
                words.append(_close(cur, x, y0, y1))
                cur = None
            continue
        if cur is None:
            cur = [ch, x]
        else:
            cur[0] += ch
    if cur is not None:
        words.append(_close(cur, line_box[2], y0, y1))
    return words


def _close(cur, x1, y0, y1):
    text, x0 = cur
    # A space can start a little left of the last glyph's left edge only if the model
    # emitted it inside a word; keep the box non-degenerate in that case.
    x1 = max(x1, x0 + 1)
    return {"text": text, "box": [round(x0, 2), y0, round(x1, 2), y1]}
