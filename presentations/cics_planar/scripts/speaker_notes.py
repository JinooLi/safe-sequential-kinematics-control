"""Read keyed speaker notes, including nested TeX groups in their text."""
import re


def read_speaker_notes(path):
    source = path.read_text()
    notes = {}
    for match in re.finditer(r"^\\SpeakerNote\{([^{}]+)\}\{(\d+)\}\{", source, re.M):
        key, seconds = match[1], int(match[2])
        start = cursor = match.end()
        depth = 1
        while cursor < len(source) and depth:
            char = source[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError(f"Unclosed speaker note: {key}")
        if key in notes:
            raise ValueError(f"Duplicate speaker note: {key}")
        notes[key] = (seconds, source[start:cursor-1].strip())
    return notes
