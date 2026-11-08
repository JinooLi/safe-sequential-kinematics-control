"""Export speaker-notes.tex in the slide order defined by main.tex."""
from pathlib import Path
import re
import json
from speaker_notes import read_speaker_notes

root = Path(__file__).resolve().parents[1]
source = (root / "main.tex").read_text()
plan = json.loads((root / "evidence/story_plan.json").read_text())
blocks = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", source, re.S)
notes = read_speaker_notes(root / "speaker-notes.tex")
output = ["석사 학위 발표 대본 — 20분, 한국어 발표 / 영어 슬라이드", "",
          f"본문 {plan['main_slides']}장(표지와 마무리 포함), 보충 자료 {plan['appendix_slides']}쪽.",
          "목표 발화 시간: 19분 + 화면 전환과 호흡 1분 = 총 20분.",
          "시간은 리허설 기준이며 실제 말하는 속도에 맞게 조정합니다.",
          "아래 영어 제목은 슬라이드와 대응하기 위한 표기이며, 낭독할 대본은 한국어입니다.", ""]
elapsed = 0
count = 0
used = set()
def stamp(s):
    return f"{s // 60:02d}:{s % 60:02d}"
for block in blocks:
    options = re.match(r"\[([^]]*)\]", block)
    label = re.search(r"(?:^|,)label=([^,]+)", options[1]) if options else None
    if label is None:
        continue
    key = label[1]
    assert key in notes, f"Missing speaker note: {key}"
    assert key not in used, f"Duplicate frame label: {key}"
    used.add(key)
    count += 1
    seconds, text = notes[key]
    title = re.match(r"(?:\[[^]]*\])?\{([^{}]*)\}", block)
    title = title[1] if title and title[1] else next(s['title'] for s in plan['slides'] if s['label'] == key)
    output.extend([f"{count:02d}. {title}",
                   f"{stamp(elapsed)}–{stamp(elapsed + seconds)} / {seconds}초",
                   text, ""])
    elapsed += seconds
assert count == plan["main_slides"] and elapsed == plan["speech_seconds"] == 1140
assert used == set(notes), "Speaker notes without matching slides"
(root / "발표대본.txt").write_text("\n".join(output) + "\n")
print(f"Exported {count} notes; {elapsed} seconds + 60 seconds buffer.")
