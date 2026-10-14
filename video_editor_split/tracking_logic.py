
def target_matches(target_text, detected_text):
    return bool(target_text) and target_text in detected_text


def should_stop_search(lost_count, patience):
    return lost_count >= patience


def clamp_roi(x, y, w, h, frame_width, frame_height):
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2 - x1, y2 - y1)
