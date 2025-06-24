import json
from fastapi import FastAPI

app = FastAPI()

@app.post("/feedback")
async def save_feedback(
    instruction: str,
    generated_code: str,
    rating: int,  # 1-5 점
    corrected_code: str = None
):
    """사용자 피드백을 학습 데이터로 저장"""
    
    if rating >= 4:  # 좋은 평가를 받은 코드만 저장
        training_entry = {
            "instruction": instruction,
            "input": "",
            "output": corrected_code if corrected_code else generated_code
        }
        
        # 학습 데이터에 추가
        try:
            with open("auto_training_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        
        data.append(training_entry)
        
        with open("auto_training_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {"status": "success", "message": "학습 데이터에 추가됨"}
    
    return {"status": "skipped", "message": "낮은 평가로 인해 스킵됨"}