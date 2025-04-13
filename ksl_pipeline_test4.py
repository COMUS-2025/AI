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

    return result


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
        return {
            "error": "수어 문장 추출 실패",
            "original": sentence,
            "ksl_transformed": result1
        }
    
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
        "final_result": "\n".join(final_output)
    }



# Test the pipeline with example sentences
# This part tests the pipeline with various Korean sentences.
# It prints the results in a readable format.
if __name__ == "__main__":
    word_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ksl_dictionary_words_augmented.csv")
    dict_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ksl_dictionary_final_re.csv")
    test_sentences = [
            "오늘은 진짜 재밌었어요! 학교 끝나고 친구랑 놀이터에서 숨바꼭질도 하고, 엄마가 좋아하는 떡볶이도 사줘서 최고였어요.",
            # "조금 피곤했어요. 숙제가 많아서 집에 와서도 계속 공부했거든요. 그래도 다 하고 나니까 뿌듯했어요.",
            # "괜찮았어요! 특별한 일은 없었지만, 점심에 나온 돈가스가 맛있어서 기분 좋았어요.",
            # "기분이 설렜어요! 오늘 체육 수업 있는 날이라 아침부터 운동화도 닦고 준비 열심히 했어요.",
            # "졸렸어요… 눈이 안 떠져서 엄마가 세 번이나 깨워줬어요. 학교 가기 싫었지만 갔더니 괜찮았어요.",
            # "조금 걱정됐어요. 수학 시험 보는 날이었거든요. 시험 생각하면서 계속 머릿속으로 계산했어요.",
            # "점심시간이요! 친구들이랑 같이 밥 먹고 수다 떨 때 제일 신나요. 먹고 나서 놀기도 하거든요.",
            # "학교 끝나는 순간이요! 그때가 제일 자유로워요. 집 가서 간식 먹을 생각에 늘 들떠요.",
            # "아침에 학교 가서 친구들 만나는 순간이요. 어제 못 한 얘기 빨리 하고 싶어서요!",
            # "좋아하는 노래 크게 틀고 따라 부르면 기분이 좋아져요. 방에서 혼자 콘서트 하는 느낌이에요.",
            # "강아지 인형 꼭 안고 있으면 마음이 편해져요. 진짜 말은 못 해도 위로받는 기분이에요.",
            # "엄마한테 가서 안겨 있으면 금방 나아져요. 그냥 말 없이 있어도 마음이 따뜻해져요.",
            # "숙제요. 국어랑 수학 둘 다 있어서 두 시간 넘게 했어요. 중간에 졸릴 뻔했어요.",
            # "학교에서 앉아서 수업 듣는 시간이 제일 길었어요. 칠판 보다가 멍도 많이 때렸어요.",
            # "그림 그린 시간이요! 미술 시간에 풍경화 그렸는데, 하늘 색 고르는 데만 20분은 걸렸어요.",
            # "쉬는 시간에 친구랑 카드놀이 했는데 제가 이겼어요! 친구가 놀라서 웃었고 저도 기분 좋았어요.",
            # "집에 와서 엄마가 피자 시켜준다고 했을 때요! 피자 먹을 생각에 춤췄어요.",
            # "선생님이 갑자기 우리에게 퀴즈 내주셨는데, 제가 제일 먼저 맞혔어요! 칭찬도 받고 사탕도 받았어요.",
            # "조용히 다가가서 '무슨 일 있어?' 하고 물어봐요. 말 안 해도 옆에 있어주면 조금은 괜찮아지는 것 같아요.",
            # "간식 나눠줘요. 맛있는 거 먹으면 기분 좀 나아지니까요. 초콜릿 주면 웃는 친구도 있었어요.",
            # "같이 그림 그리자고 하거나 게임하자고 해요. 말로 위로하는 건 어려운데 같이 놀면 좀 웃게 돼요.",
            # "먼저 웃으면서 '안녕! 난 지희야. 너 이름 뭐야?' 하고 말 걸어요. 그러면 금방 친해질 수 있어요.",
            # "같이 놀자고 먼저 말해요! 놀이터나 쉬는 시간에 '우리 같이 놀래?' 하면 친구가 생겨요.",
            # "그 친구가 좋아하는 걸 같이 해보는 것도 좋아요. 그 친구가 좋아하는 그림 그리기 같이 하면서 얘기 나눠요.",
            # "너무너무 반가워요! 마치 생일 선물 받은 것처럼 기분이 막 들뜨고 신나요.",
            # "어색하지만 빨리 예전처럼 놀고 싶다는 생각 들어요. 그래서 장난치면서 다시 친해져요.",
            # "얼른 놀고 싶은 마음에 웃음이 막 나와요. 만나자마자 '너 뭐하고 지냈어?' 하고 폭풍 질문해요.",
            # "연필 떨어뜨렸을 때 주워주는 거요. 사소하지만 친구가 고마워해요.",
            # "친구가 숙제 몰라서 힘들어하면 제가 아는 부분 알려줘요. 같이 하면 금방 끝나니까요.",
            # "기분 안 좋아 보일 때 좋아하는 스티커 하나 몰래 책상 위에 올려줘요. 기분이 좋아졌으면 좋겠어요.",
            # "같이 게임하거나 숨바꼭질할 때요! 뛰어다니면서 소리 지르면 웃음이 멈추질 않아요.",
            # "급식 먹으면서 수다 떨 때요. 웃긴 얘기 하다가 우유 뿜을 뻔했어요.",
            # "같이 숙제하거나 프로젝트할 때요. 서로 도와가면서 하니까 재미있고 뿌듯해요.",
            # "직접 만든 팔찌요! 구슬로 알록달록하게 만들면 친구가 좋아할 것 같아요.",
            # "예쁜 편지지에 손편지 써서 몰래 가방에 넣어주는 거요. 깜짝 놀라고 기뻐할 거예요.",
            # "귀여운 캐릭터 스티커나 지우개요! 문구점에서 사면 비싸지 않고 친구가 좋아해요.",
            # "도서관이요! 조용하고 책도 많고, 가끔 만화책도 있어서 시간이 금방 가요.",
            # "운동장이요! 마음껏 뛰어놀 수 있어서 제일 좋아요. 축구도 하고 줄넘기도 해요.",
            # "미술실이요! 물감이랑 색종이도 많고 만들기 할 때 너무 재밌어요.",
            # "글쓰기 시간에 제 이야기 쓴 거 보고 선생님이 '정말 생생하게 잘 썼다!'고 칭찬해주셨어요.",
            # "수학 문제 다 맞았을 때 선생님이 '대단해~ 계산이 빠르구나!' 해서 기분이 엄청 좋았어요.",
            # "친구 도와줬을 때 선생님이 '참 착하구나~' 하고 말해주셨어요. 얼굴이 막 빨개졌어요.",
            # "오늘 과학 시간에 선생님이 문제 내주셨는데, 내가 제일 먼저 맞혔어요! 친구들이 우와~ 하면서 박수 쳐줘서 진짜 뿌듯했어요.",
            # "체육 시간에 줄넘기 대회 했는데, 나 혼자 100개 넘게 해서 상 받았어요! 엄마한테 자랑하려고 상장 가방에 꼭 넣어왔어요.",
            # "도서관에서 친구한테 책 찾는 거 도와줬는데, 고맙다고 인사해서 기분이 진짜 좋았어요. 내가 누군가한테 도움이 됐다는 게 자랑스러웠어요.",
            # "작년에 담임이셨던 민정 선생님이요! 항상 웃으시고 내가 슬플 때 몰래 손편지도 주셔서 아직도 기억나요.",
            # "미술 선생님이요! 색칠할 때 자유롭게 하라고 해주셔서 제일 신났어요. 제 그림 칭찬도 많이 해주셨거든요.",
            # "이번 학기 수학 선생님이요. 문제 어려울 때마다 차근차근 설명해주셔서 수학이 재미있어졌어요.",
            # "저는 발표요! 앞에 나가서 말할 때 떨리긴 해도 또박또박 말 잘한다고 선생님이 칭찬해주셨어요.",
            # "그림 잘 그리는 거요. 친구들이 자기 책 표지 만들어달라고 부탁할 때마다 제가 도와줘요.",
            # "체육이요! 축구랑 달리기 진짜 잘해서 운동장에 나가면 항상 친구들이 팀에 들어오라고 해요.",
            # "오늘은 재밌었어요! 친구랑 같이 점심도 먹고, 체육 시간에 배드민턴도 해서 신났어요.",
            # "조금 힘들었어요. 수학 시험 봤는데 헷갈리는 문제가 나와서 걱정돼요. 그래도 열심히 했어요.",
            # "오늘은 비 와서 실내에서 놀았는데, 친구들이랑 퀴즈 놀이해서 웃음이 멈추질 않았어요!",
            # "바다 가고 싶어요! 수영도 하고 조개도 줍고, 밤엔 모래 위에서 누워서 별 보고 싶어요.",
            # "산속 캠핑이요! 텐트 치고 고기도 구워 먹고, 밤에 손전등 들고 산책하는 거 해보고 싶어요.",
            # "놀이공원이요! 가족이랑 같이 롤러코스터 타고 솜사탕도 먹고 사진도 많이 찍고 싶어요.",
            # "같이 놀자고 자주 말해요. 특히 엄마랑 보드게임 하고 싶을 때 자꾸 부르게 돼요.",
            # "배고프다고 매일 말하는 것 같아요. 밥 먹는 시간이 제일 기다려져요!",
            # "아침마다 학교 가기 전에 꼭 사랑한다고 말해요. 가족이 웃어줄 때 기분 최고에요.",
            # "서로의 이야기를 잘 들어주는 게 제일 중요한 것 같아요. 제가 말할 때 눈 보고 들어주면 기분 좋아요.",
            # "같이 노는 시간이 많아야 돼요! 놀면서 서로 기분이나 생각을 자연스럽게 알 수 있어요.",
            # "화났을 때 바로 화내지 말고, 왜 그런지 먼저 물어보면 더 잘 이해할 수 있을 거예요.",
            # "학교에서 속상한 일 있었는데 엄마가 제 얘기 다 들어주고 같이 초콜릿 먹으면서 위로해줬어요. 너무 고마웠어요.",
            # "아빠가 밤 늦게까지 일하시는데도 제가 숙제 도와달라고 하면 항상 도와주셨어요. 그게 진짜 감사했어요.",
            # "언니가 제 생일 때 몰래 손편지 써줘서 울 뻔했어요. 항상 싸우는데 그날은 진짜 감동이었어요.",
            # "명절에 할머니 집 가서 온 가족이 둘러앉아 송편 만들 때가 특별해요. 이야기하면서 웃는 게 좋아요.",
            # "밤에 영화 보면서 같이 이불 덮고 간식 먹을 때요! 그 시간이 너무 따뜻하고 편해요.",
            # "생일날 가족이 다 같이 케이크 불 꺼주고 노래 불러줄 때 제일 특별해요. 주인공이 된 느낌이에요.",
            # "엄마가 바빠서 제 얘기 못 듣고 좀 이따 말하라고 했을 때 서운했어요. 그냥 그 순간 들어줬으면 좋겠었어요.",
            # "아빠가 화나서 큰소리로 말했을 때 무서웠어요. 나쁜 일 한 건 아니었는데 오해했을 때 속상했어요.",
            # "제가 잘했는데 혼난 적이 있어서 울었어요. 그땐 진짜 너무 억울했어요… 나중엔 미안하다고 해주셨지만요.",
            # "여름에 친구들이랑 물총 싸움도 하고, 바닷가 가서 튜브 타고 놀고 싶어요.",
            # "봄에 벚꽃 날릴 때 엄마랑 도시락 싸서 공원에서 소풍 가고 싶어요. 사진도 예쁘게 찍고요.",
            # "겨울에 눈 오면 눈사람 만들고 썰매 타고, 코코아 마시면서 따뜻한 이불 속에 들어가고 싶어요.",
            # "동물병원에서 일하는 수의사요! 아픈 강아지랑 고양이들 고쳐주고 싶어요. 동물이 너무 좋아요.",
            # "유튜버요! 게임하면서 재밌게 말도 하고, 나만의 방송 만들어서 사람들 웃게 하고 싶어요.",
            # "우주비행사요! 진짜 우주에 가서 별이랑 지구도 직접 보고 싶어요. 우주복 입고 떠나는 상상 자주 해요.",
            # "레고 조립할 때요! 한 번 시작하면 어느새 두 시간도 훅 지나가 있어서 깜짝 놀라요.",
            # "친구들이랑 놀이터에서 놀 때요. 뛰어다니고 놀다 보면 해가 져 있어서 시간 가는 줄 몰라요.",
            # "그림 그릴 때요. 색칠하고 꾸미다 보면 너무 재밌어서 시간이 순식간에 사라져요.",
            # "열기구 타보고 싶어요! 하늘 위에서 아래 풍경 보면 얼마나 신기할지 상상만 해도 짜릿해요.",
            # "수족관에서 하룻밤 자는 체험이요! 물고기랑 상어 보면서 자는 건 진짜 멋질 것 같아요.",
            # "뮤지컬 무대에 서보는 거요! 무대 위에서 노래도 부르고 춤도 추면 짜릿할 것 같아요.",
            # "아침부터 저녁까지 게임하고 싶어요! 과자 먹으면서 내가 제일 좋아하는 RPG 게임 할래요.",
            # "가족이랑 하루 종일 여행 가고 싶어요. 맛있는 거 먹고 사진도 찍고, 호텔에서 수영도 하고요.",
            # "책방 가서 좋아하는 만화책 마음껏 읽고, 원하는 책 다 골라서 가방 가득 담아보고 싶어요.",
            # "우리 반에서 새로 전학 온 친구랑 빨리 친해지고 싶다고 자주 생각해요. 말 걸고 싶은데 쑥스러워요.",
            # "강아지 키우고 싶은 마음이 계속 들어요. 유튜브에서도 강아지 영상 자꾸만 찾아봐요.",
            # "언제쯤 마스크 안 쓰고 맘껏 친구랑 놀 수 있을까 자주 생각해요. 예전처럼 놀고 싶어요.",
        ]
     
    with open("ksl_pipeline_test_results.json", "w", encoding="utf-8") as f:
        f.write("[\n")  

        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n예문 {i}: {sentence}")
            try:
                result = run_pipeline(sentence, word_csv_path, dict_csv_path)
            except Exception as e:
                result = {
                    "original": sentence,
                    "error": f"처리 중 오류 발생: {str(e)}"
                }

            print("원문:", result.get("original", "[원문 없음]"))
            print("수어 문장:", result.get("ksl_sentence", "없음"))
            print("설명:", result.get("ksl_explanation", "없음"))
            print("의미 확장 결과:", result.get("semantic_matched", "없음"))
            print("최종 결과 (링크 포함):\n" + result.get("final_result", "없음"))

            json.dump(result, f, ensure_ascii=False, indent=2)

            if i != len(test_sentences):
                f.write(",\n")
            else:
                f.write("\n")

        f.write("]\n")  