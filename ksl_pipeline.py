import openai
import os
import json
import re
from dotenv import load_dotenv
from konlpy.tag import Okt
import csv
import pandas as pd

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize Korean tokenizer
okt = Okt()

# Korean sentence analysis function
# This function analyzes a Korean sentence and extracts its grammatical components.
def analyze_korean_sentence(sentence):
    tags = {
        "문형": "의문문" if sentence.strip().endswith("?") else "평서문",
        "주어": None,
        "목적어": None,
        "서술어": None,
        "동사종류": None,
        "수식어": [],
        "부사어": [],
        "의문사": [],
        "부정표현": None,
        "복합문_접속사": [],
        "감정": None,
        "문장성분순": [],
        "비수지표지": {"문장 끝": "{휴지}"}
    }

    tokens = okt.pos(sentence, norm=True, stem=True)

    for word, tag in tokens:
        if tag == "Noun" and not tags["주어"]:
            tags["주어"] = word
        elif tag == "Noun" and tags["서술어"] and not tags["목적어"]:
            tags["목적어"] = word
        elif tag in ["Verb", "Adjective"] and not tags["서술어"]:
            tags["서술어"] = word
        elif tag == "Adjective":
            tags["수식어"].append(word)
        elif tag == "Adverb":
            tags["부사어"].append(word)
        elif tag == "Determiner":
            tags["의문사"].append(word)
        elif word in ["안", "못", "않다", "아니다", "없다", "말다", "불가능", "안되다"]:
            tags["부정표현"] = word
        elif word in ["그리고", "하지만", "그래서", "그러나"]:
            tags["복합문_접속사"].append(word)

    tags["문장성분순"] = [k for k in ["주어", "수식어", "부사어", "의문사", "목적어", "서술어"] if tags[k]]

    if not tags["주어"]:
        tags["주어"] = "너" if tags["문형"] == "의문문" else "나"

    타동사 = ["읽다", "보다", "사다", "입다", "먹다", "치다", "받다", "도와주다", "알려주다"]
    if tags["서술어"]:
        tags["동사종류"] = "타동사" if any(v in tags["서술어"] for v in 타동사) else "자동사"

    return tags

# Korean Sign Language prompt construction
# This function builds the system and user prompts for the KSL translation task.
def build_signlang_prompt(sentence, tags):
    system_prompt = """
[역할 부여]
너는 전문 수어 번역가야. 

[목표 설명]
사용자가 입력한 한국어 문장을 한국수어 문법에 맞게 정확히 변환해야 해. 주어진 문장을 수어 문법에 맞게 재구성하고, 변환 이유도 함께 설명해줘.

[규칙 설정]
다음 규칙을 반드시 지켜야 해:
1. 어순 재배열:
    - 기본 구조는 [주어] [목적어] [서술어]
    - 형용사/자동사 문장은 [주어] [서술어]
    - 수식어는 피수식어 앞에, 단 명사 수식은 명사 뒤에
    - 부사어는 서술어 바로 뒤 또는 문장 앞/끝
    - 의문사는 문장 마지막
2. 문장 성분 보완:
    - 생략된 주어가 있을 경우: 평서문은 [나], 의문문은 [너]
    - 반복되는 성분은 앞/뒤 문장에서 가져와 보완
3. 부정문 처리:
    - 서술어 뒤에 정확한 수어 부정 표현 사용 
    - 부정법을 구현할 때 ‘안’, ‘못’, ‘말-’과 같은 부정 요소 대신 [아니다], [못하다], [말다]와 같은 부정어를 활용하여 부정법을 구현할 것
    - [아니다]: 모두 어떤 사실을 단순 부정할 때 사용
    - [없다]: 상태, 성질, 존재 등을 부정할 때 사용
    - [못하다]: 능력 부재, 외부적 요인 등으로 할 수 없음을 나타낼 때 사용
    - 이중 부정은 제거
4. 복합문 처리:
    - 접속사는 문장 분할로 처리하고 {휴지} 삽입
    - 각 문장 단위에 주어-목적어-서술어 구조 유지
5. 감정 및 문형:
    - 감정 표현은 {감정: 기쁨} 형태로
    - 의문문은 비수지표지 설명 없이 어순만 반영
6. 표현 방식:
    - 조사는 제거하되 의미는 유지
    - 서술어 중심 표현 유지
    - 반복 명사는 [a], [b]로 대체
    - {휴지}는 필요한 문장 구획에만 삽입하고 성분 사이에는 넣지 마.
    - '~고 싶다'의 표현은 [원하다]로 대체

[예시 제공]
예시1: 
- 입력: 가방이 무거워 이상하다고 생각해 확인해 보니 가방이 바뀌었다.
- 출력: [가방바뀌다] [가방들고오다] [무겁다] [이상하다] [가방 보다] [아차]
예시2: 
- 입력: 엄마가 전에 여동생한테 ‘게으름 피우지 말고 공부 열심히 하라’고 말씀하셨다.
- 출력: [엄마] [과거] [여동생] [말하다] {휴지} [게으름] [말고] [공부] [열심]
예시3: 
- 입력: 오늘 내가 주번이어서 우유 급식을 반으로 가져와야 했는데 가져오다가 복도에 엎어서 다 흘려서 민망했다.
- 출력: [오늘] [나] [주번] [책임] {휴지} [우유] [반] [운반] [해야 하다] {휴지} [복도] [운반하다] [엎지르다] {휴지} [우유] [바닥] [쏟아지다] {휴지} [부끄럽다]

[출력 형식]
결과는 반드시 다음 포맷을 따라줘.
[수어 문법 변환 결과]: ...
[설명]: 주요 변환 규칙 적용 설명 (어순, 부정, 접속사, 감정 등)
    """.strip()

    user_prompt = f"""
다음 문장을 수어 문법에 맞게 바꿔줘.

[입력 문장]: "{sentence}"
[문장 분석 결과]: {json.dumps(tags, ensure_ascii=False)}

[출력 형식]:
[수어 문법 변환 결과]: ...
[설명]: ...
    """.strip()
    return system_prompt, user_prompt


def load_dictionary_csv(csv_path):
    groups = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            group = list(set([item.strip() for item in row if item.strip()]))
            if group:
                groups.append(group)
    return groups
    
# Semantic matching prompt construction
# This function builds the system and user prompts for the semantic matching task.
# It uses the CSV file to find semantically similar words.
def build_semantic_prompt(sign_sentence, word_csv_groups):
    word_csv_groups = "\n".join([f"- {', '.join(group)}" for group in word_csv_groups])

    system_prompt = """
[역할 부여]
너는 단어 의미 분석에 특화된 언어학자야. 모든 한국어 단어의 의미, 어형 변화, 문맥 속 쓰임을 잘 이해하고 있어.

[목표 설명]
내가 준 수어 문법 기반 문장에는 [ ] 안에 단어들이 들어 있어. 이 단어들을 내가 가진 csv 목록에 포함된 단어들로 가능한 많이 바꿔줘. 그래야 이 단어들에 수어 영상 링크를 연결할 수 있어.

[규칙 설정]
다음 규칙을 반드시 지켜야 해:
1. 너는 내가 제공한 csv 안의 단어 그룹들만 사용할 수 있어.
2. csv의 각 행은 같은 의미 또는 기능을 갖는 단어들의 그룹이야
3. 입력된 단어가 csv 그룹에 없는 경우라도, 의미나 형태가 유사한 단어가 있다면 csv에 포함된 표현으로 바꿔줘.
4. [단어] 형식을 유지하면서 바꾸고, 치환된 단어는 *를 붙여줘.
4. 치환이 불가능하면 원래 단어를 그대로 두고, * 표시도 하지 마.
5. 각 단어는 csv 그룹에 따라 [단어1, 단어2, ...] 형태로 치환해줘 (중복 제거).
6. {휴지} 같은 특수 표시는 그대로 유지해.
7. 치환된 단어가 중복되면 한 번만 출력해줘.

[예시 제공]
예시1:
- 입력: [가방바뀌다] [가방들고오다] [무겁다] [이상하다] [가방 보다] [아차]
- 출력: *[가방,들다] *[교환,바꾸다,맞바꾸다] *[가방,들다] *[묵직하다,무겁다] *[가방] *[보다] *[아차]

예시2:
- 입력: [엄마] [과거] [여동생] [말하다] {휴지} [게으름] [말고] [공부] [열심]
- 출력: *[어머니,모친,어미,엄마] *[과거,전,지나다] *[여동생,누이,누이동생] *[말,말하다,언어] {휴지} *[게으름,게으르다,게으름을 피우다,나태,태만] [말고] *[공부,학업] *[근면,열성,열심,꾸준하다,부지런하다]

예시3:
- 입력: [오늘] [나] [주번] [책임] {휴지} [우유] [반] [운반] [해야 하다] {휴지} [복도] [운반하다] [엎지르다] {휴지} [우유] [바닥] [쏟아지다] {휴지} [부끄럽다]
- 출력: *[오늘,금일,이번,오늘날,현재] *[자신,나,저,내] [주번] *[감당,사명,책임,담당,소임,역임]

[단어 리스트 제공]
다음은 바꿀 수 있는 단어 그룹이야. 이 목록 안에서만 변환할 수 있어: {word_csv_groups}

[출력 형식]
결과는 반드시 다음 포맷을 따라줘.
[의미 매칭 결과]: ...
    """.strip()
    
    user_prompt = f"""
아래 수어 문장을 단어 단위로 의미가 같은 단어로 바꿔줘. CSV에 있는 단어들만 사용해줘.

[입력 문장(수어 문법 변환 결과)]: {sign_sentence}

[출력 형식]:
[의미 매칭 결과]: ...
    """.strip()
    return system_prompt, user_prompt


# OpenAI API call function
# This function calls the OpenAI API to get the KSL translation and semantic matching results.
def call_gpt(system_prompt, user_prompt, model="gpt-4o"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"[API 호출 실패] {str(e)}"

def extract_sign_sentence(gpt_output):
    match = re.search(r"\[수어 문법 변환 결과\]:\s*(.+)", gpt_output)
    return match.group(1).strip() if match else ""

def extract_sign_explanation(gpt_output):
    match = re.search(r"\[설명\]:\s*(.+)", gpt_output, re.DOTALL)
    return match.group(1).strip() if match else ""

# KSL links and video URL attachment function
# This function attaches KSL dictionary links and video URLs to the matched sentence.
# It uses a CSV file to find the corresponding URLs for the matched words.
def attach_ksl_links(matched_sentence, dict_csv_path, word_csv_path):
    import pandas as pd
    import re

    # Load KSL dictionary
    dict_df = pd.read_csv(dict_csv_path)
    word_to_url = {}
    for _, row in dict_df.iterrows():
        for word in str(row['단어']).split("/"):
            word = word.strip()
            if word:
                word_to_url[word] = row['수어 동영상 URL (MP4)']

    # Load original/augmented synonym CSV
    synonym_df = pd.read_csv(word_csv_path)  
    word_to_original = {}
    for _, row in synonym_df.iterrows():
        original_group = row['original']
        augmented_group = row['augmented']
        originals = [w.strip() for w in original_group.split(",") if w.strip()]
        augmented = [w.strip() for w in augmented_group.split(",") if w.strip()]
        if originals:
            representative = originals[0]  # 기준 단어
            for word in augmented:
                word_to_original[word] = representative

    # Match words in the sentence
    result = []
    tokens = re.findall(r"\*?\[.*?\]|\{.*?\}", matched_sentence)

    for token in tokens:
        if token.startswith("{"):
            result.append(token)
            continue

        is_starred = token.startswith("*")
        content = token.lstrip("*").strip("[]")
        candidates = [w.strip() for w in content.split(",")]

        found = None
        for w in candidates:
            if w in word_to_url:
                found = (w, word_to_url[w])
                break
            elif w in word_to_original:
                original = word_to_original[w]
                if original in word_to_url:
                    found = (original, word_to_url[original])
                    break

        if found:
            word, url = found
            display = f"*[{word}] {url}" if is_starred else f"[{word}]  {url}"
        else:
            display = token
        result.append(display)

    return "\n".join(result)


# Main pipeline function
# This function orchestrates the entire pipeline from sentence analysis to KSL translation and semantic matching.
# It takes a sentence and a CSV file path as input and returns the results.
def run_pipeline(sentence, word_csv_path, dict_csv_path):
    tags = analyze_korean_sentence(sentence)
    sys1, user1 = build_signlang_prompt(sentence, tags)
    result1 = call_gpt(sys1, user1)
    sign_sentence = extract_sign_sentence(result1)
    sign_explanation = extract_sign_explanation(result1)
    if not sign_sentence:
        return {"error": "수어 문장 추출 실패", "ksl_transformed": result1}
    
    word_groups = load_dictionary_csv(word_csv_path)
    sys2, user2 = build_semantic_prompt(sign_sentence, word_groups)
    result2 = call_gpt(sys2, user2)
    final_output = attach_ksl_links(result2, dict_csv_path, word_csv_path)
    
    return {
        "original": sentence,
        "ksl_sentence": sign_sentence,
        "ksl_explanation": sign_explanation,
        # "ksl_transformed_full": result1,
        "semantic_matched": result2,
        "final_result": final_output
    }
