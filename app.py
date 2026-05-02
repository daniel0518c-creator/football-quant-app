import streamlit as st
import pandas as pd
import psycopg2
import requests
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ================= 1. 页面 UI 初始化 =================
st.set_page_config(page_title="AI 足球量化投注看板", page_icon="⚽", layout="wide")
st.title("⚽ 智能足球量化投注看板 (全欧联赛版)")
st.markdown("通过结合 **Elo 战力模型** 与 **The Odds API 实时盘口**，自动寻找预期收益率 (EV) 大于 0 的价值投注机会。")

# 从云端安全环境中读取秘密 Key
DB_URI = st.secrets["DB_URI"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

# ================= 2. 核心算法与模型缓存 =================
# 使用缓存机制：防止每次刷新网页都要重新算一遍9000场比赛
@st.cache_resource(show_spinner="⏳ 正在加载历史数据并唤醒 AI 大脑，请稍候...")
def load_and_train_model():
    conn = psycopg2.connect(DB_URI)
    query = "SELECT competition, match_date, home_team, away_team, home_score, away_score FROM matches ORDER BY match_date ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # 计算 Elo
    def get_expected_score(ra, rb): return 1 / (1 + 10 ** ((rb - ra) / 400))
    elo_dict = {}
    home_elo, away_elo = [], []
    
    for _, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        if ht not in elo_dict: elo_dict[ht] = 1500
        if at not in elo_dict: elo_dict[at] = 1500
            
        cur_ht_elo, cur_at_elo = elo_dict[ht], elo_dict[at]
        home_elo.append(cur_ht_elo)
        away_elo.append(cur_at_elo)
        
        # 简化版 Elo 更新
        ea = get_expected_score(cur_ht_elo, cur_at_elo)
        actual_a = 1 if row['home_score'] > row['away_score'] else (0.5 if row['home_score'] == row['away_score'] else 0)
        elo_dict[ht] = cur_ht_elo + 20 * (actual_a - ea)
        elo_dict[at] = cur_at_elo + 20 * ((1 - actual_a) - (1 - ea))
        
    df['home_elo'] = home_elo
    df['away_elo'] = away_elo
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    
    def get_outcome(row):
        if row['home_score'] > row['away_score']: return 'HomeWin'
        elif row['home_score'] == row['away_score']: return 'Draw'
        else: return 'AwayWin'
    df['outcome'] = df.apply(get_outcome, axis=1)

    X = df[['home_elo', 'away_elo', 'elo_diff']]
    y = df['outcome']
    
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    
    return model, model.classes_, elo_dict

# 唤醒 AI 大脑
model, classes, elo_dict = load_and_train_model()

# ================= 3. 博彩数学计算器 =================
def calculate_ev(prob, odds): return (prob * odds) - 1
def calculate_kelly(prob, odds, fraction=0.25):
    b = odds - 1
    kelly = (prob * b - (1 - prob)) / b
    return max(0, kelly * fraction)

# ================= 4. 实时盘口侦察与页面展示 =================
st.sidebar.header("⚙️ 侦察设置")
selected_league = st.sidebar.selectbox("选择要侦察的联赛:", 
                                     ["soccer_epl (英超)", "soccer_spain_la_liga (西甲)", "soccer_uefa_champs_league (欧冠)"])
league_key = selected_league.split(" ")[0]

if st.sidebar.button("🚀 一键预测今日赛事"):
    st.subheader("📡 实时盘口侦察报告")
    url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h"
    
    with st.spinner('正在连线欧洲博彩公司获取实时赔率...'):
        resp = requests.get(url)
        
    if resp.status_code == 200:
        matches = resp.json()
        if not matches:
            st.info("🤷 目前该联赛没有即将开打的比赛盘口。")
        else:
            for match in matches:
                # 提取数据
                home_team, away_team = match['home_team'], match['away_team']
                try:
                    bookmaker = match['bookmakers'][0]
                    home_odds = next(item['price'] for item in bookmaker['markets'][0]['outcomes'] if item['name'] == home_team)
                except:
                    continue # 找不到赔率则跳过
                
                # AI 计算
                ht_elo = elo_dict.get(home_team, 1500)
                at_elo = elo_dict.get(away_team, 1500)
                
                features = pd.DataFrame({'home_elo': [ht_elo], 'away_elo': [at_elo], 'elo_diff': [ht_elo - at_elo]})
                probs = model.predict_proba(features)[0]
                home_win_prob = probs[list(classes).index('HomeWin')]
                
                ev = calculate_ev(home_win_prob, home_odds)
                kelly = calculate_kelly(home_win_prob, home_odds)
                
                # 渲染 UI 卡片
                with st.container():
                    st.markdown(f"### {home_team} (主) vs {away_team}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label=f"机构主胜赔率 ({bookmaker['title']})", value=f"{home_odds}")
                    col2.metric(label="AI 算出真实主胜率", value=f"{home_win_prob*100:.1f}%")
                    
                    if ev > 0:
                        col3.success(f"🔥 发现价值 (EV: +{ev*100:.1f}%)")
                        st.info(f"💰 **系统建议下注仓位**: 总本金的 **{kelly*100:.2f}%** 买主胜")
                    else:
                        col3.error(f"❌ 无投资价值 (EV: {ev*100:.1f}%)")
                    st.divider()
    else:
        st.error(f"获取 API 数据失败: {resp.status_code}")
