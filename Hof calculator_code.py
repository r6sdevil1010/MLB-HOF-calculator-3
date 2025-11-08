import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------
# ⚾ 기본 데이터/모델 학습 파트
# -----------------------------
# (예시용 - 실제 데이터셋 연결 시 교체 가능)
data = pd.DataFrame({
    "WAR": [50, 65, 75, 80, 100],
    "HOFm": [90, 110, 130, 140, 180],
    "JAWS": [45, 55, 60, 70, 85],
    "Elected": [0, 0, 1, 1, 1]
})

# 득표율/헌액 확률용 간단 모델
model_vote = LogisticRegression()
model_prob = LogisticRegression()

X = data[["WAR", "HOFm", "JAWS"]]
y = data["Elected"]
model_vote.fit(X, y)
model_prob.fit(X, y)

# -----------------------------
# ⚙️ 유틸 함수
# -----------------------------
def simulate_vote_growth(start_vote):
    """연차별 득표율 예측 (BBWAA 트렌드 반영)"""
    votes = [start_vote]
    for i in range(1, 10):
        inc = 0.05 + 0.08 * (1 - votes[-1] / 100)
        votes.append(min(100, votes[-1] * (1 + inc)))
    return votes


def predict_HOF(name, WAR, HOFm, JAWS, doping=False, leadership=0.5, influence=0.5, era_adjust=0.0):
    """명전 확률 + 득표율 예측 통합 함수"""
    basic_vote = model_vote.predict_proba([[WAR, HOFm, JAWS]])[0, 1] * 100
    basic_prob = model_prob.predict_proba([[WAR, HOFm, JAWS]])[0, 1]

    # 외부 요인 반영
    ext_factor = (-0.35 if doping else 0) + leadership * 0.15 + influence * 0.2 + era_adjust * 0.1
    final_vote = max(0, min(100, basic_vote * (1 + ext_factor)))
    final_prob = max(0, min(1, basic_prob * (1 + ext_factor)))

    # 연차별 시뮬레이션
    vote_trend = simulate_vote_growth(final_vote)

    return {
        "name": name,
        "basic_vote": basic_vote,
        "final_vote": final_vote,
        "basic_prob": basic_prob,
        "final_prob": final_prob,
        "vote_trend": vote_trend
    }


def summarize_result(res):
    text = f"⚾ {res['name']} — Hall of Fame 예측 결과\n\n"
    text += f"📊 기본모델 득표율: {res['basic_vote']:.1f}%\n"
    text += f"🏅 외부요인 반영 득표율: {res['final_vote']:.1f}%\n"
    text += f"🎯 헌액 확률(성적기반): {res['basic_prob'] * 100:.1f}%\n"
    text += f"💬 최종 헌액 확률(외부요인 반영): {res['final_prob'] * 100:.1f}%\n\n"
    text += f"📈 연차별 득표율 추정: {[round(v, 1) for v in res['vote_trend']]}"
    return text


# -----------------------------
# 🌐 Streamlit UI
# -----------------------------
st.title("⚾ MLB Hall of Fame 예측 시스템")
st.caption("WAR, HOFm, 리더십, 도핑 여부 등을 고려한 명전 확률 추정기")

name = st.text_input("선수 이름", "Joe Mauer")
WAR = st.number_input("WAR", 0.0, 150.0, 65.0)
HOFm = st.number_input("HOF Monitor 점수", 0.0, 300.0, 120.0)
JAWS = st.number_input("JAWS 점수", 0.0, 100.0, 55.0)
doping = st.checkbox("도핑 이력 있음", value=False)
leadership = st.slider("리더십/영향력 점수", 0.0, 1.0, 0.5)
influence = st.slider("커리어/문화적 영향력", 0.0, 1.0, 0.5)
era_adjust = st.slider("시대 보정 (타고투저/투고타저)", -0.3, 0.3, 0.0)

if st.button("예측 실행"):
    res = predict_HOF(name, WAR, HOFm, JAWS, doping, leadership, influence, era_adjust)
    st.text(summarize_result(res))
