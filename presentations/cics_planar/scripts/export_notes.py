"""별도 한국어 대본과 페이지별 시간표 생성."""
from pathlib import Path
import re
import json
from speaker_notes import read_speaker_notes

root = Path(__file__).resolve().parents[1]
source = (root / "main.tex").read_text().split(r"\appendix", 1)[0]
notes = read_speaker_notes(root / "speaker-notes.tex")
frames = re.findall(r"\\begin\{frame\}(.*?)\\end\{frame\}", source, re.S)
slides, used = [], set()
elapsed = 0
lines = ["CICS 발표 대본 — 한국어 슬라이드 / 한국어 발표, 총 10분", "",
         "본문 18장, 목표 발화 9분 20초 + 화면 전환·호흡 여유 40초.",
         "페이지별 시간은 리허설에 맞춰 조정. 부록은 질의응답용.", ""]
outline = ["CICS 발표 구성 — 제출 원고의 3-DOF planar 내용", "",
           "문제 → CLF–CBF QP와 한계 → 관련 연구와 연구 질문 → IK에서 link 목표 구성 → 순차 제어 → 효과와 한계", ""]
def stamp(seconds):
    return f"{seconds//60:02}:{seconds%60:02}"

for i, block in enumerate(frames, 1):
    key = re.search(r"label=([^,\]]+)", block)[1]
    assert key in notes and key not in used
    used.add(key)
    seconds, note = notes[key]
    match = re.match(r"\[[^\]]*\]\{([^{}]*)\}", block)
    title = match[1] if match and match[1] else {"title": "표지", "thanks": "감사합니다"}[key]
    lines.extend([f"{i:02}. {title}", f"{stamp(elapsed)}–{stamp(elapsed+seconds)} / {seconds}초", note, ""])
    subtitle = re.search(r"\\framesubtitle\{([^{}]*)\}", block)
    slides.append({"page": i, "label": key, "title": title, "seconds": seconds,
                   "paper_section": subtitle[1] if subtitle else None})
    outline.append(f"{i:02}. {title} — {seconds}초")
    elapsed += seconds
assert used == set(notes) and len(slides) == 18 and elapsed == 560
(root / "발표대본.txt").write_text("\n".join(lines) + "\n")
(root / "발표구성.txt").write_text("\n".join(outline) + "\n")
(root / "evidence/story_plan.json").write_text(json.dumps({"main_slides":18,"speech_seconds":elapsed,
    "buffer_seconds":600-elapsed,"slides":slides},ensure_ascii=False,indent=2)+"\n")
print(f"{len(slides)} notes; {elapsed} s speech + {600-elapsed} s buffer.")
