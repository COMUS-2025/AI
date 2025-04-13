from flask import Flask, request
from ksl_pipeline import run_pipeline
import os
import re
import traceback
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
word_csv_path = os.path.join(BASE_DIR, "ksl_dictionary_words_augmented.csv")
dict_csv_path = os.path.join(BASE_DIR, "ksl_dictionary_final_re.csv")

@app.route('/ksl/translate', methods=['POST'])
def translate_to_ksl():
    try: 
        data = request.get_json()
        if not data or "sentence" not in data:
            return app.response_class(
                response=json.dumps({
                    "status": 400, 
                    "message": "문장이 제공되지 않았습니다."
                }, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )

        sentence = data["sentence"]
        result = run_pipeline(sentence, word_csv_path, dict_csv_path)
        
        if "error" in result:
            return app.response_class(
                response=json.dumps({
                    "status": 500, 
                    "message": "KSL 변환 실패", 
                    "detail": result["error"]
                }, ensure_ascii=False),
                status=500,
                mimetype='application/json'
            )

        # Extracting the KSL sentence 
        converted_sentence = []
        for token in result["ksl_sentence"].split():
            token = token.strip("[]")
            if token:
                converted_sentence.append(token)

        # Extracting the KSL video URLs 
        ksl_videos = {}
        lines = result["final_result"].split("\n")

        for orig_word in converted_sentence:
            pattern = re.compile(rf"\*\[([^\]]*?\b{re.escape(orig_word)}\b[^\]]*?)\]")
            for line in lines:
                match = pattern.search(line)
                if match:
                    candidates = [w.strip() for w in match.group(1).split(",")]
                    for w in candidates:
                        url = extract_ksl_url(w, result["final_result"])
                        if url:
                            ksl_videos[orig_word] = {
                                "video_url": url
                            }
                            break
                    break
       
        # response format               
        response_payload = {
            "status": 200,
            "message": "요청이 성공했습니다.",
            "data": {
                "converted_sentence": converted_sentence,
                "ksl_videos": ksl_videos
            }
        }
        # Adding debug information if in DEBUG_MODE
        if DEBUG_MODE:
            response_payload["debug_data"] = {
                "ksl_sentence": result.get("ksl_sentence"),
                "semantic_matched": result.get("semantic_matched"),
                "final_result": result.get("final_result")
            }
            
        return app.response_class(
            response=json.dumps(response_payload, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        traceback.print_exc()
        return app.response_class(
            response=json.dumps({
                "status": 500,
                "message": "서버 오류",
                "detail": str(e)
            }, ensure_ascii=False),
            status=500,
            mimetype='application/json'
        )

def extract_ksl_url(word, final_result):
    if not word:
        return ""
    pattern = re.compile(rf"\*?\[{re.escape(word)}\]\s+(https?://[^\s]+)")
    match = pattern.search(final_result)
    return match.group(1) if match else ""



if __name__ == "__main__":
     app.run(host='0.0.0.0', port=443, ssl_context=("/app/server.crt", "/app/server.key"))
